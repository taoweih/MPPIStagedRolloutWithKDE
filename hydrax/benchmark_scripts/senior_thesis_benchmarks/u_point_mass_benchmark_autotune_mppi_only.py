import mujoco

from hydrax.algs import MPPI, MPPIStagedRollout, MPPIMemoryContinuous
from hydrax.simulation.deterministic import run_interactive, run_benchmark

from hydrax.tasks.u_point_mass import UPointMass

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from hydrax import ROOT
from datetime import datetime
from pathlib import Path
import os
import json
import time

from ray import tune
from mujoco import mjx
from flax import nnx
import optax


# def _sync_tree(x):
#     return jax.tree_util.tree_map(
#         lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x
#     )


def run_benchmark_memory_upoint(
    controller: MPPIMemoryContinuous,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    initial_knots: jax.Array = None,
    GOAL_THRESHOLD: float = 1.0,
    num_trials=100,
):
    """Run benchmark for MPPIMemoryContinuous which requires global_memory (neural net weights)."""
    print(
        f"Using controller {type(controller).__name__}\n"
        f"Planning with {controller.ctrl_steps} steps "
        f"over a {controller.plan_horizon} second horizon "
        f"with {controller.num_knots} knots."
    )

    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep} Hz"
    )

    mjx_model = controller.task.model
    mjx_data = mjx.make_data(mjx_model)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    policy_params = controller.init_params(initial_knots=initial_knots)

    _jit_optimize = jax.jit(controller.optimize)
    def jit_optimize(mjx_data, policy_params, global_memory=None):
        policy_params, rollouts, rollout_states, global_memory = _jit_optimize(mjx_data, policy_params, global_memory)
        if hasattr(controller, 'params'):
            controller.params = policy_params
        return policy_params, rollouts, global_memory
    jit_interp_func = jax.jit(controller.interp_func)

    # Pre-train the neural network on true cost landscape
    global_memory = controller.nn_weight_template

    num_starting = 100000
    grid_inputs = jax.random.uniform(jax.random.PRNGKey(52), (num_starting, 2), minval=-1.0, maxval=1.0)

    def get_true_cost(qpos):
        d = mjx.make_data(controller.task.model)
        d = d.replace(qpos=qpos)
        d = mjx.kinematics(controller.task.model, d)
        return controller.task.terminal_cost(d)

    grid_targets = jax.vmap(get_true_cost)(grid_inputs)

    model = nnx.merge(controller.graphdef, global_memory, controller.static_state)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-3), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, inputs, targets, optimizer):
        def loss_fn(m):
            preds = m(inputs).squeeze()
            return jnp.mean((preds - targets) ** 2)
        grads = nnx.grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss_fn(model)

    print("Pre-training neural network...")
    for i in range(3000):
        loss = train_step(model, grid_inputs, grid_targets, optimizer)
        if i % 100 == 0:
            print(f"  Iter {i}, Loss: {loss:.4f}")

    _, global_memory, _ = nnx.split(model, nnx.Param, ...)

    # Warm-up the controller
    print("Jitting the controller...")
    st = time.time()
    policy_params, rollouts, _ = jit_optimize(mjx_data, policy_params, global_memory)
    policy_params, rollouts, _ = jit_optimize(mjx_data, policy_params, global_memory)

    tq = jnp.arange(0, sim_steps_per_replan) * mj_model.opt.timestep
    tk = policy_params.tk
    knots = policy_params.mean[None, ...]
    _ = jit_interp_func(tq, tk, knots)
    _ = jit_interp_func(tq, tk, knots)
    print(f"Time to jit: {time.time() - st:.3f} seconds")

    num_success = 0
    total_iteration = 0
    number_of_iteration = 1000
    number_of_trials = num_trials
    total_plan_time = 0
    total_plan_steps = 0
    freq = 0

    state_trajectories = np.zeros((number_of_trials, number_of_iteration) + mj_data.qpos.shape)
    control_trajectories = np.zeros((number_of_trials, number_of_iteration) + mj_data.ctrl.shape)

    base_seed = 5
    mj_data_reset = mujoco.MjData(mj_model)
    pretrained_global_memory = global_memory  # save pre-trained weights

    for i in range(number_of_trials):
        mj_data.qpos[:] = mj_data_reset.qpos
        mj_data.qvel[:] = mj_data_reset.qvel
        mj_data.mocap_pos[:] = mj_data_reset.mocap_pos
        mj_data.mocap_quat[:] = mj_data_reset.mocap_quat
        mj_data.time = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        policy_params = controller.init_params(initial_knots=initial_knots, seed=base_seed + i)
        global_memory = pretrained_global_memory  # reset NN weights to pre-trained state
        reached_goal = False

        for j in range(number_of_iteration):
            start_time = time.time()

            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
                time=mj_data.time,
            )

            plan_start = time.time()
            policy_params, rollouts, global_memory = jit_optimize(mjx_data, policy_params, global_memory=global_memory)
            plan_time = time.time() - plan_start
            total_plan_time += plan_time
            total_plan_steps += 1
            freq = (total_plan_steps / total_plan_time)

            sim_dt = mj_model.opt.timestep
            t_curr = mj_data.time
            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            tk = policy_params.tk
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]

            for k in range(sim_steps_per_replan):
                mj_data.ctrl[:] = np.array(us[k])
                mujoco.mj_step(mj_model, mj_data)
                if controller.task.success_function(mj_data, mj_data.ctrl[:]) < GOAL_THRESHOLD:
                    reached_goal = True
                    break

            state_trajectories[i][j] = mj_data.qpos
            control_trajectories[i][j] = mj_data.ctrl

            if reached_goal:
                num_success += 1
                total_iteration += j
                break

            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            rtr = step_dt / (time.time() - start_time)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s",
                end="\r",
            )

    avg_success_iteration = 0 if num_success == 0 else total_iteration / num_success
    print("")
    return num_success, freq, state_trajectories, control_trajectories, avg_success_iteration

# Need to be wrapped in main loop for async simulation
if __name__ == "__main__":
    special_name = "thesis"

    task = UPointMass()
    goal_threshold = 0.4

    # sweep settings
    NUM_TRIALS = 50

    Horizon_steps = 20
    Horizon_start = 0.5
    Horizon_end = 2.0

    Sample_steps = 20
    Sample_start = 128
    Sample_end = 4096
    samples_list = np.linspace(Sample_start, Sample_end, Sample_steps).astype(int)

    # manually set common parameters (not auto-tuned)
    SPLINE_TYPE = "zero"
    NUM_SAMPLES = 512
    NOISE_LEVEL = 3.0  
    TEMPERATURE = 0.001 

    # tune_horizon = 1.5
    # tune_cpu_num = 3
    # tune_gpu_num = 0

    # # tune MPPI num_knots
    # def objective_mppi(config):
    #     ctrl = MPPI(task, num_samples=NUM_SAMPLES, noise_level=NOISE_LEVEL, temperature=TEMPERATURE,
    #                 plan_horizon=tune_horizon, spline_type=SPLINE_TYPE, num_knots=config["num_knots"])
    #     mj_model = task.mj_model
    #     mj_model.opt.timestep = 0.01
    #     mj_data = mujoco.MjData(mj_model)
    #     num_success, _, _, _, _ = run_benchmark(
    #             ctrl,
    #             mj_model,
    #             mj_data,
    #             frequency=50,
    #             GOAL_THRESHOLD=goal_threshold,
    #             num_trials=10,
    #         ) 
    #     return {"score": num_success}

    # search_space = { 
    #     "num_knots": tune.choice(list(range(2, 32, 2))),
    # }

    # tuner = tune.Tuner(tune.with_resources(objective_mppi, {"cpu": tune_cpu_num, "gpu": tune_gpu_num}), param_space=search_space, tune_config= tune.TuneConfig(num_samples=3, metric="score", mode="max")) 
    # results = tuner.fit()

    # # MPPI specific
    # MPPI_NUM_KNOTS = results.get_best_result().config["num_knots"]

    # # tune MPPI staged rollout: num_knots, kde_bandwidth, num_knots_per_stage
    # def objective_staged(config):
    #     ctrl = MPPIStagedRollout(task, num_samples=NUM_SAMPLES, noise_level=NOISE_LEVEL, temperature=TEMPERATURE,
    #                 kde_bandwidth=config["kde_bandwidth"], num_knots_per_stage=config["num_knots_per_stage"],
    #                 plan_horizon=tune_horizon, spline_type=SPLINE_TYPE, num_knots=config["num_knots"])
    #     mj_model = task.mj_model
    #     mj_model.opt.timestep = 0.01
    #     mj_data = mujoco.MjData(mj_model)
    #     num_success, _, _, _, _ = run_benchmark(
    #             ctrl,
    #             mj_model,
    #             mj_data,
    #             frequency=50,
    #             GOAL_THRESHOLD=goal_threshold,
    #             num_trials=10,
    #         ) 
    #     return {"score": num_success}

    # search_space = { 
    #     "num_knots": tune.choice([8, 16, 32]),
    #     "kde_bandwidth": tune.uniform(0, 1),
    #     "num_knots_per_stage": tune.choice([2, 4, 8]),
    # }

    # tuner = tune.Tuner(tune.with_resources(objective_staged, {"cpu": tune_cpu_num, "gpu": tune_gpu_num}), param_space=search_space, tune_config= tune.TuneConfig(num_samples=3, metric="score", mode="max")) 
    # results = tuner.fit()

    # # MPPI staged rollout specific
    # STAGED_NUM_KNOTS = results.get_best_result().config["num_knots"]
    # NUM_KNOTS_PER_STAGE = results.get_best_result().config["num_knots_per_stage"]
    # KDE_BANDWIDTH = results.get_best_result().config["kde_bandwidth"]

    MPPI_NUM_KNOTS = 16
    STAGED_NUM_KNOTS = 16
    NUM_KNOTS_PER_STAGE = 4
    KDE_BANDWIDTH = 0.15
    MEMORY_NUM_KNOTS = 16


    
    NUM_CONTROLLERS = 4
    ctrl_names = ["MPPI", "MPPI Density", "MPPI Memory", "MPPI Density + Memory"]

    success = np.zeros((NUM_CONTROLLERS, Horizon_steps))
    success_time = np.zeros((NUM_CONTROLLERS, Horizon_steps))
    all_frequency = np.zeros((NUM_CONTROLLERS, Horizon_steps))
    all_state_trajectory = [[] for _ in range(NUM_CONTROLLERS)]
    all_control_trajectory = [[] for _ in range(NUM_CONTROLLERS)]

    def state_selection_function(state: mjx.Data) -> jax.Array:
        return state.qpos[..., 0:2]

    horizons = np.linspace(Horizon_start, Horizon_end, Horizon_steps)

    for h in tqdm(range(Horizon_steps)):
        HORIZON = float(horizons[h])

        ctrl_list = [MPPI(task, num_samples=NUM_SAMPLES, noise_level=NOISE_LEVEL, temperature=TEMPERATURE,
                         plan_horizon=HORIZON, spline_type=SPLINE_TYPE, num_knots=MPPI_NUM_KNOTS), 

                    MPPIStagedRollout(task, num_samples=NUM_SAMPLES, noise_level=NOISE_LEVEL, temperature=TEMPERATURE, 
                                    num_knots_per_stage=NUM_KNOTS_PER_STAGE, plan_horizon=HORIZON, spline_type=SPLINE_TYPE,
                                    num_knots=STAGED_NUM_KNOTS, kde_bandwidth=KDE_BANDWIDTH),

                    MPPIMemoryContinuous(task, num_samples=NUM_SAMPLES, noise_level=NOISE_LEVEL, temperature=TEMPERATURE,
                                    plan_horizon=HORIZON, spline_type=SPLINE_TYPE, num_knots=MEMORY_NUM_KNOTS,
                                    state_selection_function=state_selection_function,
                                    grid_min=-1.0, grid_max=1.0, online_learning_rate=1e-3,
                                    goal_position=jnp.array([[0.025, 0.775]]),
                                    use_staged_rollout=False),

                    MPPIMemoryContinuous(task, num_samples=NUM_SAMPLES, noise_level=NOISE_LEVEL, temperature=TEMPERATURE,
                                    plan_horizon=HORIZON, spline_type=SPLINE_TYPE, num_knots=MEMORY_NUM_KNOTS,
                                    num_knots_per_stage=NUM_KNOTS_PER_STAGE,
                                    kde_bandwidth=KDE_BANDWIDTH, state_selection_function=state_selection_function,
                                    grid_min=-1.0, grid_max=1.0, online_learning_rate=1e-3,
                                    goal_position=jnp.array([[0.025, 0.775]]),
                                    use_staged_rollout=True)]
        
        for j in range(len(ctrl_list)):
            ctrl = ctrl_list[j]

            mj_model = task.mj_model
            mj_model.opt.timestep = 0.01

            mj_data = mujoco.MjData(mj_model)

            if isinstance(ctrl, MPPIMemoryContinuous):
                num_success, control_freq, state_trajectory, control_trajectory, avg_success_iteration = run_benchmark_memory_upoint(
                    ctrl,
                    mj_model,
                    mj_data,
                    frequency=50,
                    GOAL_THRESHOLD=goal_threshold,
                    num_trials=NUM_TRIALS,
                )
            else:
                num_success, control_freq, state_trajectory, control_trajectory, avg_success_iteration = run_benchmark(
                    ctrl,
                    mj_model,
                    mj_data,
                    frequency=50,
                    GOAL_THRESHOLD=goal_threshold,
                    num_trials=NUM_TRIALS,
                )

            success_time[j, h] = avg_success_iteration * 0.01
            success[j, h] = num_success / NUM_TRIALS * 100
            all_frequency[j, h] = control_freq
            all_state_trajectory[j].append(state_trajectory)
            all_control_trajectory[j].append(control_trajectory)

    # Save directory (shared by both sweeps)
    curr_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = Path(ROOT)/"benchmark"/f"u_point_mass_benchmark_{special_name}_{curr_time}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Horizon sweep: state and control trajectories
    file_path = os.path.join(save_dir, "horizon_trajectory.npz")
    all_state_trajectory = np.array(all_state_trajectory)
    all_control_trajectory = np.array(all_control_trajectory)
    np.savez(file_path, state_trajectory=all_state_trajectory, control_trajectory=all_control_trajectory)
    
    # Horizon sweep: control frequency
    file_path = os.path.join(save_dir, "horizon_frequency.csv")
    np.savetxt(file_path, all_frequency, delimiter=",",fmt="%.2e")

    plt.figure()
    for j in range(all_frequency.shape[0]):
        plt.plot(np.linspace(Horizon_start, Horizon_end, Horizon_steps), all_frequency[j], label=ctrl_names[j])
    plt.title(f'Task {type(task).__name__}')
    plt.xlabel("Horizon (s)")
    plt.ylabel("Control Frequency (HZ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"horizon_frequency.png", dpi=300)
    plt.close()

    # Horizon sweep: success rate
    file_path = os.path.join(save_dir, "horizon_success_count.csv")
    np.savetxt(file_path, success, delimiter=",",fmt="%.1f")

    plt.figure()
    for j in range(success.shape[0]):
        plt.plot(np.linspace(Horizon_start, Horizon_end, Horizon_steps), success[j], label=ctrl_names[j])
    plt.title(f'Task {type(task).__name__}')
    plt.xlabel("Horizon (s)")
    plt.ylabel("Success Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"horizon_success_count.png", dpi=300)
    plt.close()

    file_path = os.path.join(save_dir, "horizon_success_time.csv")
    np.savetxt(file_path, success_time, delimiter=",",fmt="%.4f")

    plt.figure()
    x_horizon = np.linspace(Horizon_start, Horizon_end, Horizon_steps)
    for j in range(success.shape[0]):
        mask = success_time[j] != 0
        if np.any(mask):
            plt.plot(x_horizon[mask], success_time[j][mask], label=ctrl_names[j])
    plt.title(f'Task {type(task).__name__}')
    plt.xlabel("Horizon (s)")
    plt.ylabel("Average success time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"horizon_success_time.png", dpi=300)
    plt.close()

    # ==================== NUM_SAMPLES SWEEP ====================
    SWEEP_HORIZON = 1.2  # fixed horizon for the samples sweep

    Sample_steps = len(samples_list)

    success_samples = np.zeros((NUM_CONTROLLERS, Sample_steps))
    success_time_samples = np.zeros((NUM_CONTROLLERS, Sample_steps))
    all_frequency_samples = np.zeros((NUM_CONTROLLERS, Sample_steps))
    all_state_trajectory_samples = [[] for _ in range(NUM_CONTROLLERS)]
    all_control_trajectory_samples = [[] for _ in range(NUM_CONTROLLERS)]

    for s in tqdm(range(Sample_steps)):
        n_samples = int(samples_list[s])

        ctrl_list_samples = [
            MPPI(task, num_samples=n_samples, noise_level=NOISE_LEVEL, temperature=TEMPERATURE,
                 plan_horizon=SWEEP_HORIZON, spline_type=SPLINE_TYPE, num_knots=MPPI_NUM_KNOTS),

            MPPIStagedRollout(task, num_samples=n_samples, noise_level=NOISE_LEVEL, temperature=TEMPERATURE,
                              num_knots_per_stage=NUM_KNOTS_PER_STAGE, plan_horizon=SWEEP_HORIZON,
                              spline_type=SPLINE_TYPE, num_knots=STAGED_NUM_KNOTS, kde_bandwidth=KDE_BANDWIDTH),

            MPPIMemoryContinuous(task, num_samples=n_samples, noise_level=NOISE_LEVEL, temperature=TEMPERATURE,
                                 plan_horizon=SWEEP_HORIZON, spline_type=SPLINE_TYPE, num_knots=MEMORY_NUM_KNOTS,
                                 state_selection_function=state_selection_function,
                                 grid_min=-1.0, grid_max=1.0, online_learning_rate=1e-3,
                                 goal_position=jnp.array([[0.025, 0.775]]),
                                 use_staged_rollout=False),

            MPPIMemoryContinuous(task, num_samples=n_samples, noise_level=NOISE_LEVEL, temperature=TEMPERATURE,
                                 plan_horizon=SWEEP_HORIZON, spline_type=SPLINE_TYPE, num_knots=MEMORY_NUM_KNOTS,
                                 num_knots_per_stage=NUM_KNOTS_PER_STAGE,
                                 kde_bandwidth=KDE_BANDWIDTH, state_selection_function=state_selection_function,
                                 grid_min=-1.0, grid_max=1.0, online_learning_rate=1e-3,
                                 goal_position=jnp.array([[0.025, 0.775]]),
                                 use_staged_rollout=True)]

        for j in range(len(ctrl_list_samples)):
            ctrl = ctrl_list_samples[j]

            mj_model = task.mj_model
            mj_model.opt.timestep = 0.01

            mj_data = mujoco.MjData(mj_model)

            if isinstance(ctrl, MPPIMemoryContinuous):
                num_success, control_freq, state_trajectory, control_trajectory, avg_success_iteration = run_benchmark_memory_upoint(
                    ctrl,
                    mj_model,
                    mj_data,
                    frequency=50,
                    GOAL_THRESHOLD=goal_threshold,
                    num_trials=NUM_TRIALS,
                )
            else:
                num_success, control_freq, state_trajectory, control_trajectory, avg_success_iteration = run_benchmark(
                    ctrl,
                    mj_model,
                    mj_data,
                    frequency=50,
                    GOAL_THRESHOLD=goal_threshold,
                    num_trials=NUM_TRIALS,
                )

            success_time_samples[j, s] = avg_success_iteration * 0.01
            success_samples[j, s] = num_success / NUM_TRIALS * 100
            all_frequency_samples[j, s] = control_freq
            all_state_trajectory_samples[j].append(state_trajectory)
            all_control_trajectory_samples[j].append(control_trajectory)

    # Samples sweep: state and control trajectories
    file_path = os.path.join(save_dir, "samples_trajectory.npz")
    all_state_trajectory_samples = np.array(all_state_trajectory_samples)
    all_control_trajectory_samples = np.array(all_control_trajectory_samples)
    np.savez(file_path, state_trajectory=all_state_trajectory_samples, control_trajectory=all_control_trajectory_samples)

    # Samples sweep: control frequency
    file_path = os.path.join(save_dir, "samples_frequency.csv")
    np.savetxt(file_path, all_frequency_samples, delimiter=",", fmt="%.2e")

    plt.figure()
    for j in range(all_frequency_samples.shape[0]):
        plt.plot(samples_list, all_frequency_samples[j], label=ctrl_names[j])
    plt.title(f'Task {type(task).__name__}')
    plt.xlabel("Number of Samples")
    plt.ylabel("Control Frequency (HZ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"samples_frequency.png", dpi=300)
    plt.close()

    # Samples sweep: success rate
    file_path = os.path.join(save_dir, "samples_success_count.csv")
    np.savetxt(file_path, success_samples, delimiter=",", fmt="%.1f")

    plt.figure()
    for j in range(success_samples.shape[0]):
        plt.plot(samples_list, success_samples[j], label=ctrl_names[j])
    plt.title(f'Task {type(task).__name__}')
    plt.xlabel("Number of Samples")
    plt.ylabel("Sucess Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"samples_success_count.png", dpi=300)
    plt.close()

    file_path = os.path.join(save_dir, "samples_success_time.csv")
    np.savetxt(file_path, success_time_samples, delimiter=",", fmt="%.4f")

    plt.figure()
    for j in range(success_samples.shape[0]):
        mask = success_time_samples[j] != 0
        if np.any(mask):
            plt.plot(samples_list[mask], success_time_samples[j][mask], label=ctrl_names[j])
    plt.title(f'Task {type(task).__name__}')
    plt.xlabel("Number of Samples")
    plt.ylabel("Average success time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"samples_success_time.png", dpi=300)
    plt.close()

    # Save combined params.json for both sweeps
    file_path = os.path.join(save_dir, "params.json")
    params = {
        "horizon_sweep": {
            "Horizon start": Horizon_start,
            "Horizon end": Horizon_end,
            "Horizon steps": Horizon_steps,
            "Number of samples": NUM_SAMPLES,
        },
        "samples_sweep": {
            "Fixed horizon (s)": SWEEP_HORIZON,
            "Sample values": samples_list.tolist(),
        },
        "common": {
            "Noise level": NOISE_LEVEL,
            "Temperature": TEMPERATURE,
            "Spline type": SPLINE_TYPE,
            "Number of trials": NUM_TRIALS,
            "Goal threshold": goal_threshold,
        },
        "MPPI": {
            "Number of knots": MPPI_NUM_KNOTS,
        },
        "MPPI Density": {
            "Number of knots": STAGED_NUM_KNOTS,
            "Number of knots per stage": NUM_KNOTS_PER_STAGE,
            "KDE Bandwidth": KDE_BANDWIDTH,
        },
        "MPPI Memory": {
            "Number of knots": MEMORY_NUM_KNOTS,
        },
        "MPPI Density + Memory": {
            "Number of knots": MEMORY_NUM_KNOTS,
            "Number of knots per stage": NUM_KNOTS_PER_STAGE,
            "KDE Bandwidth": KDE_BANDWIDTH,
        }
    }
    with open(file_path, "w") as f:
        json.dump(params, f, indent=4)

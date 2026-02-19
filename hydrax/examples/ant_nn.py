import argparse

import mujoco

from hydrax.algs import MPPI, MPPIStagedRollout, DIAL, MPPIMemoryContinuous
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.ant import Ant

import jax
import jax.numpy as jnp

from mujoco import mjx

import time
from typing import Sequence
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')
import mujoco.viewer
import numpy as np

from hydrax.alg_base import SamplingBasedController
from hydrax import ROOT
from hydrax.utils.video import VideoRecorder

from tqdm import tqdm

from flax import nnx
import optax

def _sync_tree(x):
    return jax.tree_util.tree_map(
        lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x
    )


def run_interactive_visualize_continuous(  # noqa: PLR0912, PLR0915
    controller: SamplingBasedController,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    initial_knots: jax.Array = None,
    fixed_camera_id: int = None,
    show_traces: bool = True,
    max_traces: int = 5,
    trace_width: float = 5.0,
    trace_color: Sequence = [1.0, 1.0, 1.0, 0.1],
    reference: np.ndarray = None,
    reference_fps: float = 30.0,
    record_video: bool = False,
) -> None:
    
    # Report the planning horizon in seconds for debugging
    print(
        f"Planning with {controller.ctrl_steps} steps "
        f"over a {controller.plan_horizon} second horizon "
        f"with {controller.num_knots} knots."
    )

    # Figure out how many sim steps to run before replanning
    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep} Hz"
    )

    # bench_planner(controller, mj_model,  mj_data)

    # Initialize the controller
    mjx_model = controller.task.model
    mjx_data  = mjx.make_data(mjx_model)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    policy_params = controller.init_params(initial_knots=initial_knots)
    ### Wrap this to update param stored in controller
    _jit_optimize = jax.jit(controller.optimize)
    def jit_optimize(mjx_data, policy_params, global_memory=None):
        policy_params, rollouts, rollout_states, global_memory= _jit_optimize(mjx_data, policy_params, global_memory)
        if hasattr(controller, 'params'):
            controller.params = policy_params
        return policy_params, rollouts, rollout_states, global_memory
    jit_interp_func = jax.jit(controller.interp_func)

    # Warm-up the controller
    print("Jitting the controller...")
    st = time.time()
    policy_params, rollouts, _, _= jit_optimize(mjx_data, policy_params)
    policy_params, rollouts, _, _= jit_optimize(mjx_data, policy_params)

    tq = jnp.arange(0, sim_steps_per_replan) * mj_model.opt.timestep
    tk = policy_params.tk
    knots = policy_params.mean[None, ...]
    _ = jit_interp_func(tq, tk, knots)
    _ = jit_interp_func(tq, tk, knots)
    print(f"Time to jit: {time.time() - st:.3f} seconds")
    num_traces = min(rollouts.controls.shape[1], max_traces)

    # Ghost reference setup
    if reference is not None:
        ref_data = mujoco.MjData(mj_model)
        assert reference.shape[1] == mj_model.nq
        ref_data.qpos[:] = reference[0, :]
        mujoco.mj_forward(mj_model, ref_data)

        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Transparent.
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC  # only show dynamic bodies

    # Initialize video recording if enabled
    recorder = None
    if record_video:
        # Video dimensions
        width, height = 720, 480
        # Create the video recorder
        recorder = VideoRecorder(
            output_dir=os.path.join(ROOT, "recordings"),
            width=width,
            height=height,
            fps=actual_frequency,
        )
        # Ensure model visual offscreen buffer is compatible with video recording
        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        if not recorder.start():
            record_video = False
        renderer = mujoco.Renderer(mj_model, height=height, width=width)


    # Start the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        if fixed_camera_id is not None:
            # Set the custom camera
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        # Free camera: start bottom-left, goal top-right, slightly top-down
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        # Center between start (0,0) and goal (6,6)
        viewer.cam.lookat[:] = np.array([3.0, 3.0, 0.6])

        # More top-down but still oblique
        viewer.cam.azimuth = 45.0        # makes +x go right, +y go up (typical)
        viewer.cam.elevation = -55.0     # steeper top-down (not fully top-down)
        viewer.cam.distance = 18.0       # adjust 16–22 depending on framing

        # Set up rollout traces
        if show_traces:
            num_trace_sites = len(controller.task.trace_site_ids)
            for i in range(
                num_trace_sites * num_traces * controller.ctrl_steps
            ):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=np.array(trace_color),
                )
                viewer.user_scn.ngeom += 1

        # Add geometry for the ghost reference
        if reference is not None:
            mujoco.mjv_addGeoms(
                mj_model, ref_data, vopt, pert, catmask, viewer.user_scn
            )
        cost_array = []

        ########################## Initialize global memory #########################

        global_memory = controller.nn_weight_template

        num_starting = 1000000
        grid_inputs = jax.random.uniform(jax.random.PRNGKey(52) , (num_starting, 2), minval=-10.0, maxval=10.0)


        def get_true_cost(qpos):
            d = mjx.make_data(controller.task.model)
            default_qpos = d.qpos 
            qpos = default_qpos.at[0:2].set(qpos)  # only set the xy position
            d = d.replace(qpos=qpos)
            d = mjx.kinematics(controller.task.model, d)
            return controller.task.terminal_cost(d)

        grid_targets = jax.vmap(get_true_cost)(grid_inputs)
        # jax.debug.print("grid_targets sample: {}", grid_targets[5000:5010])

        model = nnx.merge(controller.graphdef, global_memory, controller.static_state)

        optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-2), wrt=nnx.Param)

        @nnx.jit
        def train_step(model, inputs, targets, optimizer):
            def loss_fn(m):
                preds = m(inputs).squeeze()
                loss = jnp.mean((preds - targets) ** 2)
                return loss

            grads = nnx.grad(loss_fn)(model)
        
            optimizer.update(model, grads)
                
            return loss_fn(model)

        for i in range(3000):
            loss = train_step(model, grid_inputs, grid_targets, optimizer)
            if i % 10 == 0:
                print(f"Iter {i}, Loss: {loss:.4f}")

        _, global_memory, _ = nnx.split(model, nnx.Param, ...)
        
        #############################################################################

        while viewer.is_running():
        # for iter in tqdm(range(2001)): 
        
            start_time = time.time()

            # Set the start state for the controller
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
                time=mj_data.time,
            )
            
            # Do a replanning step
            plan_start = time.time()
            policy_params, rollouts, rollout_states, new_global_memory= jit_optimize(mjx_data, policy_params, global_memory=global_memory)

            if False:#iter % 200 == 0:  # plotting
                # Obstacle and goal definitions from scene.xml
                obstacle_positions = [
                    (0, 2), (2, 0), (3, 4), (2, 6),
                    (1, 3), (6, 1), (4, 3), (2, 5),
                ]
                obstacle_radius = 0.3
                goal_xy = (6.0, 6.0)
                pb = [-2.0, 10.0]  # plot bounds

                resolution = 500
                x = jnp.linspace(pb[0], pb[1], resolution)
                y = jnp.linspace(pb[0], pb[1], resolution)
                XX, YY = jnp.meshgrid(x, y)

                grid_inputs = jnp.stack([XX.flatten(), YY.flatten()], axis=1)

                eval_model = nnx.merge(controller.graphdef, global_memory, controller.static_state)
                preds = eval_model(grid_inputs).squeeze()
                image = preds.reshape(resolution, resolution)

                fig, ax = plt.subplots(figsize=(12, 12), dpi=100)

                im = ax.imshow(
                    image, origin='lower',
                    extent=[pb[0], pb[1], pb[0], pb[1]],
                    cmap="Blues", vmin=0, vmax=100
                )

                plt.colorbar(im, ax=ax, label='Predicted Cost')
                ax.set_title(f"Learned Cost Landscape (iter {iter})")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")

                # Draw obstacles (top-down view of capsules)
                for ox, oy in obstacle_positions:
                    circle = patches.Circle(
                        (ox, oy), obstacle_radius,
                        linewidth=2, edgecolor='black',
                        facecolor='gray', alpha=0.8
                    )
                    ax.add_patch(circle)

                # Draw goal
                ax.plot(goal_xy[0], goal_xy[1], 'r*', markersize=20, label='Goal', zorder=5)

                # Draw current position
                ax.plot(float(mj_data.qpos[0]), float(mj_data.qpos[1]),
                        'go', markersize=12, label='Current', zorder=5)

                # Draw sampled rollout trajectories
                try:
                    rollout_states_unpacked, nominal_trajectory_states = rollout_states
                    heuristic_states, all_states = rollout_states_unpacked

                    all_states_np = np.array(all_states.squeeze(0).squeeze(0))
                    for i in range(all_states_np.shape[0]):
                        if i % 20 == 0:
                            ax.plot(
                                all_states_np[i, :, 0],
                                all_states_np[i, :, 1],
                                color="black", linewidth=0.5, alpha=0.3
                            )

                    # Heuristic rollout
                    heuristic_np = np.array(heuristic_states.squeeze(0).squeeze(0))
                    ax.plot(heuristic_np[:, 0], heuristic_np[:, 1],
                            color="orange", linewidth=3, alpha=0.8, label='Heuristic')

                    # Nominal trajectory
                    nominal_np = np.array(nominal_trajectory_states)
                    ax.plot(nominal_np[:, 0], nominal_np[:, 1],
                            color="purple", linewidth=3, alpha=0.8, label='Nominal')
                except Exception as e:
                    print(f"Could not plot rollout states: {e}")

                ax.legend(loc='upper left')
                ax.set_xlim(pb)
                ax.set_ylim(pb)
                ax.set_aspect('equal')

                plt.savefig(f"ant_cost_landscape_{iter}.png", bbox_inches='tight')
                plt.close(fig)
                print(f"Saved ant_cost_landscape_{iter}.png")
   
            _sync_tree(policy_params)

            global_memory = new_global_memory

            plan_time = time.time() - plan_start

            # Visualize the rollouts
            if show_traces:
                ii = 0
                for k in range(num_trace_sites):
                    for i in range(num_traces):
                        for j in range(controller.ctrl_steps):
                            mujoco.mjv_connector(
                                viewer.user_scn.geoms[ii],
                                mujoco.mjtGeom.mjGEOM_LINE,
                                trace_width,
                                rollouts.trace_sites[i, j, k],
                                rollouts.trace_sites[i, j + 1, k],
                            )
                            ii += 1

            # Update the ghost reference
            if reference is not None:
                t_ref = mj_data.time * reference_fps
                i_ref = int(t_ref)
                i_ref = min(i_ref, reference.shape[0] - 1)
                ref_data.qpos[:] = reference[i_ref]
                mujoco.mj_forward(mj_model, ref_data)
                mujoco.mjv_updateScene(
                    mj_model,
                    ref_data,
                    vopt,
                    pert,
                    viewer.cam,
                    catmask,
                    viewer.user_scn,
                )

            # query the control spline at the sim frequency
            # (we assume the sim freq is the same as the low-level ctrl freq)
            sim_dt = mj_model.opt.timestep
            t_curr = mj_data.time

            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            tk = policy_params.tk
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]  # (ss, nu)

            # simulate the system between spline replanning steps
            for i in range(sim_steps_per_replan):
                mj_data.ctrl[:] = np.array(us[i])
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

                # Capture frame if recording
                if record_video and recorder.is_recording:
                    renderer.update_scene(mj_data, viewer.cam)
                    frame = renderer.render()
                    recorder.add_frame(frame.tobytes())

            cost = controller.task.success_function(mj_data, mj_data.ctrl[:])
            cost_array.append(cost)

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            # Print some timing information
            rtr = step_dt / (time.time() - start_time)
            # print(
            #     f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s",
            #     end="\r",
            # )
        plt.figure()
        plt.plot(cost_array)
        # plt.ylim(9,13)
        plt.xlabel("Iterations")
        plt.ylabel("Current objective")
        plt.title(f'Horizon: {controller.plan_horizon}s')
        plt.show()

    # Preserve the last printout
    print("")

    # Close the video recorder if recording was enabled
    if record_video and recorder is not None:
        recorder.stop()

# Need to be wrapped in main loop for async simulation
if __name__ == "__main__":

    # Define the task (cost and dynamics)
    task = Ant()

    def state_selection_function(state: mjx.Data) -> jax.Array:
        jnp_state = state.qpos[...,0:2]
        # jnp_state = state.xpos[...,1,0:2]
        # jnp_state = jnp_state.reshape(jnp_state.shape[0], -1)
        return jnp_state

    # Set up the controller
    ctrl = MPPIMemoryContinuous(
        task,
        num_samples=512,
        noise_level=0.3,
        temperature=0.01,
        num_randomizations=1,
        plan_horizon=0.2,
        spline_type="zero",
        num_knots=16,
        kde_bandwidth=0.1,
        state_selection_function= state_selection_function,

        grid_min=-10.0,
        grid_max=10.0,
        online_learning_rate = 1e-3,
        goal_position = jnp.array([[6.0, 6.0]]),
    )

    # ctrl = MPPI(
    #     task,
    #     num_samples=512,
    #     noise_level=0.4,
    #     temperature=0.01,
    #     num_randomizations=1,
    #     plan_horizon=0.2,
    #     spline_type="zero",
    #     num_knots=16,
    # )

    # Define the model used for simulation
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.01

    mj_data = mujoco.MjData(mj_model)

    run_interactive_visualize_continuous(
            ctrl,
            mj_model,
            mj_data,
            frequency=50,
            show_traces=False,
            record_video=True,
        )

    # run_interactive(
    #         ctrl,
    #         mj_model,
    #         mj_data,
    #         frequency=50,
    #         show_traces=False,
    #         record_video=False,
    #     )

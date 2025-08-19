import mujoco

from hydrax.algs import MPPI, MPPIStagedRollout, PredictiveSampling, DIAL, CEM
from hydrax.simulation.deterministic import run_interactive, run_benchmark

from hydrax.tasks.u_point_mass import UPointMass

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from hydrax import ROOT
from datetime import datetime
from pathlib import Path
import os
import json

from ray import tune

# Need to be wrapped in main loop for async simulation
if __name__ == "__main__":
    tune_horizon = 1.4
    # common non-tunable parameters
    NUM_SAMPLES = 512
    NUM_KNOTS = 16
    SPLINE_TYPE = "zero"

    # tune PS
    def objective(config):  
        ctrl = PredictiveSampling(task,num_samples=NUM_SAMPLES,noise_level=config["noise_level"]
                            ,plan_horizon=tune_horizon,spline_type=SPLINE_TYPE,num_knots=NUM_KNOTS)
        mj_model = task.mj_model
        mj_model.opt.timestep = 0.01
        mj_data = mujoco.MjData(mj_model)
        num_success, _, _, _ = run_benchmark(
                ctrl,
                mj_model,
                mj_data,
                frequency=25,
                GOAL_THRESHOLD=0.05,
                num_trials=10,
            ) 
        return {"score": num_success}

    search_space = { 
        "noise_level": tune.uniform(0, 4),
    }

    tuner = tune.Tuner(objective, param_space=search_space, tune_config= tune.TuneConfig(num_samples=10, metric="score", mode="max")) 
    results = tuner.fit()

    # common tunable parameters
    NOISE_LEVEL = results.get_best_result().config["noise_level"]

    # tune mppi
    def objective(config):  
        ctrl = MPPI(task,num_samples=NUM_SAMPLES,noise_level=NOISE_LEVEL, temperature=config["temperature"]
                            ,plan_horizon=tune_horizon,spline_type=SPLINE_TYPE,num_knots=NUM_KNOTS)
        mj_model = task.mj_model
        mj_model.opt.timestep = 0.01
        mj_data = mujoco.MjData(mj_model)
        num_success, _, _, _ = run_benchmark(
                ctrl,
                mj_model,
                mj_data,
                frequency=25,
                GOAL_THRESHOLD=0.05,
                num_trials=10,
            ) 
        return {"score": num_success}

    search_space = { 
        "tempereature": tune.uniform(0, 1),
    }

    tuner = tune.Tuner(objective, param_space=search_space, tune_config= tune.TuneConfig(num_samples=10, metric="score", mode="max")) 
    results = tuner.fit()

    # MPPI specific
    TEMPERATURE = results.get_best_result().config["temperature"]

    # tune mppi staged rollout
    def objective(config):  
        ctrl = MPPIStagedRollout(task,num_samples=NUM_SAMPLES,noise_level=NOISE_LEVEL, temperature=TEMPERATURE, kde_bandwidth=config["kde_bandwidth"]
                            ,plan_horizon=tune_horizon,spline_type=SPLINE_TYPE,num_knots=NUM_KNOTS)
        mj_model = task.mj_model
        mj_model.opt.timestep = 0.01
        mj_data = mujoco.MjData(mj_model)
        num_success, _, _, _ = run_benchmark(
                ctrl,
                mj_model,
                mj_data,
                frequency=25,
                GOAL_THRESHOLD=0.05,
                num_trials=10,
            ) 
        return {"score": num_success}

    search_space = { 
        "kde_bandwidth": tune.uniform(0, 1),
    }

    tuner = tune.Tuner(objective, param_space=search_space, tune_config= tune.TuneConfig(num_samples=10, metric="score", mode="max")) 
    results = tuner.fit()

    # MPPI staged rollout specific
    NUM_KNOTS_PER_STAGE = 4
    KDE_BANDWIDTH = results.get_best_result().config["kde_bandwidth"]

    # tune DIAL MPC
    def objective(config):  
        ctrl = DIAL(task,num_samples=NUM_SAMPLES,noise_level=NOISE_LEVEL, temperature=TEMPERATURE, beta_horizon=config["beta_horizon"], beta_opt_iter=config["beta_opt_iter"]
                            ,plan_horizon=tune_horizon,spline_type=SPLINE_TYPE,num_knots=NUM_KNOTS)
        mj_model = task.mj_model
        mj_model.opt.timestep = 0.01
        mj_data = mujoco.MjData(mj_model)
        num_success, _, _, _ = run_benchmark(
                ctrl,
                mj_model,
                mj_data,
                frequency=25,
                GOAL_THRESHOLD=0.05,
                num_trials=10,
            ) 
        return {"score": num_success}

    search_space = { 
        "beta_horizon": tune.uniform(0, 2),
        "beta_opt_iter": tune.uniform(0, 2),
    }

    tuner = tune.Tuner(objective, param_space=search_space, tune_config= tune.TuneConfig(num_samples=10, metric="score", mode="max")) 
    results = tuner.fit()

    # DIAL specific
    BETA_OPT_ITER = results.get_best_result().config["beta_opt_iter"]
    BETA_HORIZON = results.get_best_result().config["beta_horizon"]

    # tune CEM
    def objective(config):  
        ctrl = CEM(task,num_samples=NUM_SAMPLES,noise_level=NOISE_LEVEL, temperature=TEMPERATURE, explore_fraction=config["explore_fraction"], 
                   num_elites=config["num_elites"], sigma_start=config["sigma_start"], sigma_min=config["sigma_min"], plan_horizon=tune_horizon,spline_type=SPLINE_TYPE,num_knots=NUM_KNOTS)
        mj_model = task.mj_model
        mj_model.opt.timestep = 0.01
        mj_data = mujoco.MjData(mj_model)
        num_success, _, _, _ = run_benchmark(
                ctrl,
                mj_model,
                mj_data,
                frequency=25,
                GOAL_THRESHOLD=0.05,
                num_trials=10,
            ) 
        return {"score": num_success}

    search_space = { 
        "explore_fraction": tune.uniform(0, 1),
        "sigma_start": tune.uniform(NOISE_LEVEL/2, NOISE_LEVEL*2),
        "sigma_min": tune.uniform(NOISE_LEVEL/16, NOISE_LEVEL/4),
        "num_elites": tune.randint(int(NUM_SAMPLES/16), int(NUM_SAMPLES/2))
    }

    tuner = tune.Tuner(objective, param_space=search_space, tune_config= tune.TuneConfig(num_samples=10, metric="score", mode="max")) 
    results = tuner.fit()

    # CEM specific
    NUM_ELITES = results.get_best_result().config["num_elites"]
    SIGMA_START = results.get_best_result().config["sigma_start"]
    SIGMA_MIN = results.get_best_result().config["sigma_min"]
    EXPLORE_FRACTION = results.get_best_result().config["explore_fraction"]

    Horizon_steps = 25
    Horizon_start = 0.8
    Horizon_end = 2.0
    

    success = np.zeros((5, Horizon_steps)) # number of controlers by horizon
    all_frequency = np.zeros((5, Horizon_steps))
    all_state_trajectory = [[],[],[],[],[]] # number of controllers by horizon by number of iteration by number of trials by qpos shape
    all_control_trajectory = [[],[],[],[],[]]

    task = UPointMass()

    for h in tqdm(range(Horizon_steps)):
        HORIZON = (h)*0.05 + 0.8

        ctrl_list = [PredictiveSampling(task, num_samples=NUM_SAMPLES, noise_level=NOISE_LEVEL, plan_horizon=HORIZON, spline_type=SPLINE_TYPE, num_knots=NUM_KNOTS),
                     
                    MPPI(task,num_samples=NUM_SAMPLES,noise_level=NOISE_LEVEL,temperature=TEMPERATURE
                            ,plan_horizon=HORIZON,spline_type=SPLINE_TYPE,num_knots=NUM_KNOTS), 

                    MPPIStagedRollout(task, num_samples=NUM_SAMPLES, noise_level=NOISE_LEVEL, temperature=TEMPERATURE, 
                                    num_knots_per_stage=NUM_KNOTS_PER_STAGE, plan_horizon= HORIZON, spline_type=SPLINE_TYPE, num_knots=NUM_KNOTS, kde_bandwidth=KDE_BANDWIDTH),

                    DIAL(task, num_samples=NUM_SAMPLES, noise_level=NOISE_LEVEL, beta_opt_iter=BETA_OPT_ITER, beta_horizon=BETA_HORIZON, temperature=TEMPERATURE, plan_horizon=HORIZON,
                         spline_type=SPLINE_TYPE, num_knots=NUM_KNOTS),

                    CEM(task,num_samples=NUM_SAMPLES, num_elites=NUM_ELITES, sigma_start=SIGMA_START, sigma_min=SIGMA_MIN, explore_fraction=EXPLORE_FRACTION, plan_horizon=HORIZON, 
                        spline_type=SPLINE_TYPE, num_knots=NUM_KNOTS)]
        
        for j in range(len(ctrl_list)):
            ctrl = ctrl_list[j]

            mj_model = task.mj_model
            mj_model.opt.timestep = 0.01

            mj_data = mujoco.MjData(mj_model)

            num_success, control_freq, state_trajectory, control_trajectory = run_benchmark(
                ctrl,
                mj_model,
                mj_data,
                frequency=25,
                GOAL_THRESHOLD=0.05,
            )
            # num_success = h+j
            # control_freq = 0
            # state_trajectory = 0
            # control_trajectory = 0

            success[j, h] = num_success
            all_frequency[j, h] = control_freq
            all_state_trajectory[j].append(state_trajectory)
            all_control_trajectory[j].append(control_trajectory)

    #params
    curr_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = Path(ROOT)/"benchmark"/f"u_point_mass_benchmark_{curr_time}"
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(save_dir, "params.json")

    params = {
        "PS": {
            "Number of samples": NUM_SAMPLES,
            "Noise level": NOISE_LEVEL,
            "Horizon (s)": HORIZON,
            "Spline type": SPLINE_TYPE,
            "Number of knots": NUM_KNOTS
        },
        "MPPI": {
            "Number of samples": NUM_SAMPLES,
            "Noise level": NOISE_LEVEL,
            "Temperature": TEMPERATURE,
            "Horizon (s)": HORIZON,
            "Spline type": SPLINE_TYPE,
            "Number of knots": NUM_KNOTS
        },
        "MPPI staged rollout": {
            "Number of samples": NUM_SAMPLES,
            "Noise level": NOISE_LEVEL,
            "Temperature": TEMPERATURE,
            "Horizon (s)": HORIZON,
            "Spline type": SPLINE_TYPE,
            "Number of knots": NUM_KNOTS,
            "Number of knots per stage": NUM_KNOTS_PER_STAGE,
            "KDE Bandwidth": KDE_BANDWIDTH
        },
        "DIAL": {
            "Number of samples": NUM_SAMPLES,
            "Noise level": NOISE_LEVEL,
            "Temperature": TEMPERATURE,
            "Horizon (s)": HORIZON,
            "Spline type": SPLINE_TYPE,
            "Number of knots": NUM_KNOTS,
            "Beta opt iter": BETA_OPT_ITER,
            "Beta horizon": BETA_HORIZON
        },
        "CEM": {
            "Number of samples": NUM_SAMPLES,
            "Temperature": TEMPERATURE,
            "Horizon (s)": HORIZON,
            "Spline type": SPLINE_TYPE,
            "Number of knots": NUM_KNOTS,
            "Number of elites": NUM_ELITES,
            "Sigma start": SIGMA_START,
            "Sigma min": SIGMA_MIN,
            "Explore fraction": EXPLORE_FRACTION
        }
    }

    with open(file_path, "w") as f:
        json.dump(params, f, indent=4)

    # state and control trajectories
    file_path = os.path.join(save_dir, "trajectory.npz")
    all_state_trajectory = np.array(all_state_trajectory)
    all_control_trajectory = np.array(all_control_trajectory)
    np.savez(file_path, state_trajectory=all_state_trajectory, control_trajectory=all_control_trajectory)
    
    # control frequency
    file_path = os.path.join(save_dir, "frequency.csv")
    np.savetxt(file_path, all_frequency, delimiter=",",fmt="%.2e")

    plt.figure()
    for j in range(all_frequency.shape[0]):
        plt.plot(np.linspace(Horizon_start, Horizon_end, Horizon_steps), all_frequency[j], label=type(ctrl_list[j]).__name__)
    plt.title(f'Task {type(task).__name__}')
    plt.xlabel("Horizon (seconds)")
    plt.ylabel("Control Frequency (HZ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"frequency.png", dpi=300)
    plt.close()


    # sucess rate
    file_path = os.path.join(save_dir, "sucess_count.csv")
    np.savetxt(file_path, (success).astype(int), delimiter=",",fmt="%d")

    plt.figure()
    for j in range(success.shape[0]):
        plt.plot(np.linspace(Horizon_start, Horizon_end, Horizon_steps), success[j], label=type(ctrl_list[j]).__name__)
    plt.title(f'Task {type(task).__name__}')
    plt.xlabel("Horizon (seconds)")
    plt.ylabel("Sucess Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"task_{type(task).__name__}.png", dpi=300)
    plt.close()
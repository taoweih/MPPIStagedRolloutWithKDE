from ray import tune

import mujoco

from hydrax.algs import MPPI, MPPIStagedRollout, PredictiveSampling, DIAL, CEM
from hydrax.simulation.deterministic import run_interactive, run_benchmark

from hydrax.tasks.u_point_mass import UPointMass

import numpy as np
from tqdm import tqdm

from hydrax import ROOT

def auto_tune(task, controller, task_goal_threshold):
    def objective(config):  
        ctrl = controller(task,num_samples=512,noise_level=config["noise_level"],temperature=config["temperature"]
                            ,plan_horizon=1.3,spline_type="zero",num_knots=16)
        mj_model = task.mj_model
        mj_model.opt.timestep = 0.01
        mj_data = mujoco.MjData(mj_model)
        num_success, control_freq, state_trajectory, control_trajectory = run_benchmark(
                ctrl,
                mj_model,
                mj_data,
                frequency=25,
                GOAL_THRESHOLD=task_goal_threshold,
                num_trials=10,
            ) 
        return {"score": num_success}


    search_space = { 
        "noise_level": tune.uniform(0, 4),
        "temperature": tune.uniform(0, 1),
    }

    tuner = tune.Tuner(objective, param_space=search_space, tune_config= tune.TuneConfig(num_samples=30, metric="score", mode="max")) 

    results = tuner.fit()

    return results.get_best_result().config

if __name__ == "__main__":
    print(auto_tune(task=UPointMass(), controller=MPPI, task_goal_threshold=0.05))




    # tune_horizon = 1.4
    # # common non-tunable parameters
    # NUM_SAMPLES = 512
    # NUM_KNOTS = 16
    # SPLINE_TYPE = "zero"

    # # tune PS
    # def objective(config):  
    #     ctrl = PredictiveSampling(task,num_samples=NUM_SAMPLES,noise_level=config["noise_level"]
    #                         ,plan_horizon=tune_horizon,spline_type=SPLINE_TYPE,num_knots=NUM_KNOTS)
    #     mj_model = task.mj_model
    #     mj_model.opt.timestep = 0.01
    #     mj_data = mujoco.MjData(mj_model)
    #     num_success, _, _, _ = run_benchmark(
    #             ctrl,
    #             mj_model,
    #             mj_data,
    #             frequency=25,
    #             GOAL_THRESHOLD=0.05,
    #             num_trials=10,
    #         ) 
    #     return {"score": num_success}

    # search_space = { 
    #     "noise_level": tune.uniform(0, 4),
    # }

    # tuner = tune.Tuner(objective, param_space=search_space, tune_config= tune.TuneConfig(num_samples=10, metric="score", mode="max")) 
    # results = tuner.fit()

    # # common tunable parameters
    # NOISE_LEVEL = results.get_best_result().config["noise_level"]

    # # tune mppi
    # def objective(config):  
    #     ctrl = MPPI(task,num_samples=NUM_SAMPLES,noise_level=NOISE_LEVEL, temperature=config["temperature"]
    #                         ,plan_horizon=tune_horizon,spline_type=SPLINE_TYPE,num_knots=NUM_KNOTS)
    #     mj_model = task.mj_model
    #     mj_model.opt.timestep = 0.01
    #     mj_data = mujoco.MjData(mj_model)
    #     num_success, _, _, _ = run_benchmark(
    #             ctrl,
    #             mj_model,
    #             mj_data,
    #             frequency=25,
    #             GOAL_THRESHOLD=0.05,
    #             num_trials=10,
    #         ) 
    #     return {"score": num_success}

    # search_space = { 
    #     "tempereature": tune.uniform(0, 1),
    # }

    # tuner = tune.Tuner(objective, param_space=search_space, tune_config= tune.TuneConfig(num_samples=10, metric="score", mode="max")) 
    # results = tuner.fit()

    # # MPPI specific
    # TEMPERATURE = results.get_best_result().config["temperature"]

    # # tune mppi staged rollout
    # def objective(config):  
    #     ctrl = MPPIStagedRollout(task,num_samples=NUM_SAMPLES,noise_level=NOISE_LEVEL, temperature=TEMPERATURE, kde_bandwidth=config["kde_bandwidth"]
    #                         ,plan_horizon=tune_horizon,spline_type=SPLINE_TYPE,num_knots=NUM_KNOTS)
    #     mj_model = task.mj_model
    #     mj_model.opt.timestep = 0.01
    #     mj_data = mujoco.MjData(mj_model)
    #     num_success, _, _, _ = run_benchmark(
    #             ctrl,
    #             mj_model,
    #             mj_data,
    #             frequency=25,
    #             GOAL_THRESHOLD=0.05,
    #             num_trials=10,
    #         ) 
    #     return {"score": num_success}

    # search_space = { 
    #     "kde_bandwidth": tune.uniform(0, 1),
    # }

    # tuner = tune.Tuner(objective, param_space=search_space, tune_config= tune.TuneConfig(num_samples=10, metric="score", mode="max")) 
    # results = tuner.fit()

    # # MPPI staged rollout specific
    # NUM_KNOTS_PER_STAGE = 4
    # KDE_BANDWIDTH = results.get_best_result().config["kde_bandwidth"]

    # # tune DIAL MPC
    # def objective(config):  
    #     ctrl = DIAL(task,num_samples=NUM_SAMPLES,noise_level=NOISE_LEVEL, temperature=TEMPERATURE, beta_horizon=config["beta_horizon"], beta_opt_iter=config["beta_opt_iter"]
    #                         ,plan_horizon=tune_horizon,spline_type=SPLINE_TYPE,num_knots=NUM_KNOTS)
    #     mj_model = task.mj_model
    #     mj_model.opt.timestep = 0.01
    #     mj_data = mujoco.MjData(mj_model)
    #     num_success, _, _, _ = run_benchmark(
    #             ctrl,
    #             mj_model,
    #             mj_data,
    #             frequency=25,
    #             GOAL_THRESHOLD=0.05,
    #             num_trials=10,
    #         ) 
    #     return {"score": num_success}

    # search_space = { 
    #     "beta_horizon": tune.uniform(0, 2),
    #     "beta_opt_iter": tune.uniform(0, 2),
    # }

    # tuner = tune.Tuner(objective, param_space=search_space, tune_config= tune.TuneConfig(num_samples=10, metric="score", mode="max")) 
    # results = tuner.fit()

    # # DIAL specific
    # BETA_OPT_ITER = results.get_best_result().config["beta_opt_iter"]
    # BETA_HORIZON = results.get_best_result().config["beta_horizon"]

    # # tune CEM
    # def objective(config):  
    #     ctrl = CEM(task,num_samples=NUM_SAMPLES,noise_level=NOISE_LEVEL, temperature=TEMPERATURE, explore_fraction=config["explore_fraction"], 
    #                num_elites=config["num_elites"], sigma_start=config["sigma_start"], sigma_min=config["sigma_min"], plan_horizon=tune_horizon,spline_type=SPLINE_TYPE,num_knots=NUM_KNOTS)
    #     mj_model = task.mj_model
    #     mj_model.opt.timestep = 0.01
    #     mj_data = mujoco.MjData(mj_model)
    #     num_success, _, _, _ = run_benchmark(
    #             ctrl,
    #             mj_model,
    #             mj_data,
    #             frequency=25,
    #             GOAL_THRESHOLD=0.05,
    #             num_trials=10,
    #         ) 
    #     return {"score": num_success}

    # search_space = { 
    #     "explore_fraction": tune.uniform(0, 1),
    #     "sigma_start": tune.uniform(NOISE_LEVEL/2, NOISE_LEVEL*2),
    #     "sigma_min": tune.uniform(NOISE_LEVEL/16, NOISE_LEVEL/4),
    #     "num_elites": tune.randint(int(NUM_SAMPLES/16), int(NUM_SAMPLES/2))
    # }

    # tuner = tune.Tuner(objective, param_space=search_space, tune_config= tune.TuneConfig(num_samples=10, metric="score", mode="max")) 
    # results = tuner.fit()

    # # CEM specific
    # NUM_ELITES = results.get_best_result().config["num_elites"]
    # SIGMA_START = results.get_best_result().config["sigma_start"]
    # SIGMA_MIN = results.get_best_result().config["sigma_min"]
    # EXPLORE_FRACTION = results.get_best_result().config["explore_fraction"]
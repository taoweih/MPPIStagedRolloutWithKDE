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
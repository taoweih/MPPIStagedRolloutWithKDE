import argparse

import mujoco

import jax

from hydrax.algs import MPPI, MPPIStagedRollout
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.ur5e import UR5e


# Need to be wrapped in main loop for async simulation
if __name__ == "__main__":
    # jax.config.update('jax_platform_name', 'cpu')

    # Define the task (cost and dynamics)
    task = UR5e()

    # Set up the controller
    ctrl = MPPI(
        task,
        num_samples=512,
        noise_level=3.0,
        temperature=0.01,
        num_randomizations=1,
        plan_horizon=0.1,
        spline_type="zero",
        num_knots=8,
        seed=0,
    )

    # Define the model used for simulation
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.01

    mj_data = mujoco.MjData(mj_model)
    # mj_data.qpos[:] = mj_model.keyframe("home").qpos

    run_interactive(
            ctrl,
            mj_model,
            mj_data,
            frequency=50,
            show_traces=False,
            record_video=False,
            head_less=False
        )

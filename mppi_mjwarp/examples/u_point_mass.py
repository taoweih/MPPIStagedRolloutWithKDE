"""U-shaped point mass example using mujoco_warp MPPI."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mujoco
import numpy as np

from mppi_mjwarp.algs import (
    MPPI,
    MPPIStagedRollout,
    MPPIMemoryContinuous,
    MemoryPretrainConfig,
)
from mppi_mjwarp.simulation.deterministic import run_interactive
from mppi_mjwarp.tasks.u_point_mass import UPointMass


def _goal_xy(task: UPointMass) -> np.ndarray:
    data = mujoco.MjData(task.mj_model)
    mujoco.mj_forward(task.mj_model, data)
    return np.asarray(data.xpos[task.goal_pos_id, :2], dtype=np.float32)


def _memory_controller(
    task: UPointMass,
    horizon: float,
    *,
    use_staged_rollout: bool,
) -> MPPIMemoryContinuous:
    goal_xy = _goal_xy(task)

    controller = MPPIMemoryContinuous(
        task=task,
        num_samples=512,
        noise_level=3.0,
        temperature=0.001,
        plan_horizon=horizon,
        spline_type="zero",
        num_knots=16,
        iterations=1,
        seed=0,
        state_selection_function=lambda data_np: data_np["qpos"][:, 0:2],
        state_selection_fields=("qpos",),
        memory_state_dim=2,
        memory_state_min=-1.0,
        memory_state_max=1.0,
        online_learning_rate=1e-3,
        goal_state=goal_xy[None, :],
        goal_value=0.0,
        goal_weight=5000.0,
        use_staged_rollout=use_staged_rollout,
        num_knots_per_stage=4,
        kde_bandwidth=0.15,
    )

    def state_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)

    def target_function(states: np.ndarray) -> np.ndarray:
        diff = states - goal_xy[None, :]
        return 100.0 * np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)

    controller.configure_pretraining(
        state_sampler=state_sampler,
        target_function=target_function,
        config=MemoryPretrainConfig(
            sample_count=100000,
            train_steps=3000,
            batch_size=4096,
            learning_rate=1e-3,
            print_every=200,
        ),
    )
    return controller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "controller",
        nargs="?",
        default="mppi",
        choices=("mppi", "staged", "memory", "memory_staged"),
    )
    args = parser.parse_args()

    task = UPointMass()
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    show_traces = True
    record_video = False
    plan_horizon = 0.2

    if args.controller == "staged":
        controller = MPPIStagedRollout(
            task=task,
            num_samples=512,
            noise_level=3.0,
            temperature=0.001,
            num_knots_per_stage=4,
            kde_bandwidth=0.15,
            plan_horizon=plan_horizon,
            spline_type="zero",
            num_knots=16,
            iterations=1,
            seed=0,
            state_selection_function=lambda data_np: data_np["qpos"][:, 0:2],
        )
    elif args.controller == "memory":
        controller = _memory_controller(
            task=task,
            horizon=plan_horizon,
            use_staged_rollout=False,
        )
    elif args.controller == "memory_staged":
        controller = _memory_controller(
            task=task,
            horizon=plan_horizon,
            use_staged_rollout=True,
        )
    else:
        controller = MPPI(
            task=task,
            num_samples=512,
            noise_level=3.0,
            temperature=0.001,
            plan_horizon=plan_horizon,
            spline_type="zero",
            num_knots=16,
            iterations=1,
            seed=0,
        )

    run_interactive(
        controller=controller,
        mj_model=mj_model,
        mj_data=mj_data,
        frequency=50.0,
        show_traces=show_traces,
        record_video=record_video,
        max_steps=500,
    )


if __name__ == "__main__":
    main()

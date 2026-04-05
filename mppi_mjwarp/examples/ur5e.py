"""UR5e reach example using mujoco_warp MPPI."""

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
from mppi_mjwarp.tasks.ur5e import UR5e


def _goal_xyz(task: UR5e) -> np.ndarray:
    data = mujoco.MjData(task.mj_model)
    mujoco.mj_forward(task.mj_model, data)
    return np.asarray(data.xpos[task.goal_pos_id], dtype=np.float32)


def _memory_controller(
    task: UR5e,
    horizon: float,
    *,
    use_staged_rollout: bool,
) -> MPPIMemoryContinuous:
    goal_xyz = _goal_xyz(task)
    ee_site_id = task.end_effector_pos_id

    controller = MPPIMemoryContinuous(
        task=task,
        num_samples=512,
        noise_level=3.0,
        temperature=0.01,
        plan_horizon=horizon,
        spline_type="zero",
        num_knots=8,
        iterations=1,
        seed=5,
        state_selection_function=lambda data_np: data_np["site_xpos"][:, ee_site_id, :],
        state_selection_fields=("site_xpos",),
        memory_state_dim=3,
        memory_state_min=-2.0,
        memory_state_max=2.0,
        online_learning_rate=1e-3,
        goal_state=goal_xyz[None, :],
        goal_value=0.0,
        goal_weight=5000.0,
        use_staged_rollout=use_staged_rollout,
        num_knots_per_stage=2,
        kde_bandwidth=0.30,
    )

    def state_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float32)

    def target_function(states: np.ndarray) -> np.ndarray:
        diff = states - goal_xyz[None, :]
        return 10.0 * np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)

    controller.configure_pretraining(
        state_sampler=state_sampler,
        target_function=target_function,
        config=MemoryPretrainConfig(
            sample_count=500000,
            train_steps=3000,
            batch_size=4096,
            learning_rate=1e-2,
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

    task = UR5e()
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    show_traces = False
    record_video = False
    plan_horizon = 0.4

    if args.controller == "staged":
        controller = MPPIStagedRollout(
            task=task,
            num_samples=512,
            noise_level=3.0,
            temperature=0.01,
            num_knots_per_stage=2,
            kde_bandwidth=0.30,
            plan_horizon=plan_horizon,
            spline_type="zero",
            num_knots=8,
            iterations=1,
            seed=5,
            state_selection_function=(
                lambda data_np, ee_site_id=task.end_effector_pos_id: data_np[
                    "site_xpos"
                ][:, ee_site_id, :]
            ),
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
            temperature=0.01,
            plan_horizon=plan_horizon,
            spline_type="zero",
            num_knots=8,
            iterations=1,
            seed=5,
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

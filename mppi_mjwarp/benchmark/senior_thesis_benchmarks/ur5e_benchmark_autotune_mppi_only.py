"""Senior thesis benchmark sweep for UR5e in mppi_mjwarp."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from mppi_mjwarp.algs import (  # noqa: E402
    MPPI,
    MPPIStagedRollout,
    MPPIMemoryContinuous,
    MemoryPretrainConfig,
)
from mppi_mjwarp.benchmark.senior_thesis_benchmarks.benchmark_suite import (  # noqa: E402
    ControllerSpec,
    SeniorThesisBenchmarkSuite,
    SweepConfig,
)
from mppi_mjwarp.tasks import UR5e  # noqa: E402


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


def build_controller_specs() -> list[ControllerSpec]:
    return [
        ControllerSpec(
            name="MPPI",
            factory=lambda task, h: MPPI(
                task=task,
                num_samples=512,
                noise_level=3.0,
                temperature=0.01,
                plan_horizon=h,
                spline_type="zero",
                num_knots=8,
                iterations=1,
                seed=5,
            ),
        ),
        ControllerSpec(
            name="MPPI Density",
            factory=lambda task, h: MPPIStagedRollout(
                task=task,
                num_samples=512,
                noise_level=3.0,
                temperature=0.01,
                plan_horizon=h,
                spline_type="zero",
                num_knots=8,
                iterations=1,
                seed=5,
                num_knots_per_stage=2,
                kde_bandwidth=0.30,
                state_selection_function=(
                    lambda data_np, ee_site_id=task.end_effector_pos_id: data_np[
                        "site_xpos"
                    ][:, ee_site_id, :]
                ),
            ),
        ),
        ControllerSpec(
            name="MPPI Memory",
            factory=lambda task, h: _memory_controller(
                task=task,
                horizon=h,
                use_staged_rollout=False,
            ),
        ),
        ControllerSpec(
            name="MPPI Density + Memory",
            factory=lambda task, h: _memory_controller(
                task=task,
                horizon=h,
                use_staged_rollout=True,
            ),
        ),
    ]


def main() -> None:
    suite = SeniorThesisBenchmarkSuite(
        task_name="ur5e",
        task_factory=UR5e,
        controller_specs=build_controller_specs(),
        config=SweepConfig(
            horizons=np.linspace(0.1, 0.4, 3),
            num_trials=10,
            frequency=50.0,
            goal_threshold=0.4,
            max_iterations=1000,
            record_video=False,
            output_tag="thesis",
        ),
    )
    suite.run()


if __name__ == "__main__":
    main()

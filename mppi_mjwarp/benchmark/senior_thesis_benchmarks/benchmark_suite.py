"""Shared OOP benchmark suite for senior-thesis controller comparisons."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from mppi_mjwarp.simulation.deterministic import run_benchmark
from mppi_mjwarp.tasks.task_base import ROOT


@dataclass
class ControllerSpec:
    """One controller variant in the benchmark sweep."""

    name: str
    factory: Callable[[object, float], object]


@dataclass
class SweepConfig:
    """Sweep-level benchmark options."""

    horizons: Sequence[float]
    num_trials: int = 20
    frequency: float = 50.0
    goal_threshold: float = 0.5
    max_iterations: int = 1000
    record_video: bool = False
    video_trial_index: int = 0
    output_tag: str = "thesis"


class SeniorThesisBenchmarkSuite:
    """Runs the same horizon sweep across multiple controller variants."""

    def __init__(
        self,
        task_name: str,
        task_factory: Callable[[], object],
        controller_specs: Sequence[ControllerSpec],
        config: SweepConfig,
    ) -> None:
        self.task_name = task_name
        self.task_factory = task_factory
        self.controller_specs = list(controller_specs)
        self.config = config

    def run(self) -> Path:
        horizons = np.asarray(self.config.horizons, dtype=np.float32)
        num_ctrl = len(self.controller_specs)

        success = np.zeros((num_ctrl, len(horizons)), dtype=np.float32)
        success_time = np.zeros((num_ctrl, len(horizons)), dtype=np.float32)
        frequencies = np.zeros((num_ctrl, len(horizons)), dtype=np.float32)

        state_store = [[None for _ in horizons] for _ in range(num_ctrl)]
        control_store = [[None for _ in horizons] for _ in range(num_ctrl)]
        trace_store = [[None for _ in horizons] for _ in range(num_ctrl)]

        for h_idx, horizon in enumerate(horizons):
            print(f"\n=== Horizon {float(horizon):.3f}s ===")
            for c_idx, spec in enumerate(self.controller_specs):
                print(f"[{h_idx + 1}/{len(horizons)}] Running {spec.name}...")

                task = self.task_factory()
                controller = spec.factory(task, float(horizon))

                if hasattr(controller, "pretrain_memory"):
                    controller.pretrain_memory(verbose=True)

                mj_model = task.mj_model
                mj_data = mujoco.MjData(mj_model)
                mujoco.mj_forward(mj_model, mj_data)

                result = run_benchmark(
                    controller=controller,
                    mj_model=mj_model,
                    mj_data=mj_data,
                    frequency=self.config.frequency,
                    goal_threshold=self.config.goal_threshold,
                    num_trials=self.config.num_trials,
                    max_iterations=self.config.max_iterations,
                    record_video=self.config.record_video,
                    video_trial_index=self.config.video_trial_index,
                )

                success[c_idx, h_idx] = (
                    100.0 * result.num_success / self.config.num_trials
                )
                success_time[c_idx, h_idx] = (
                    result.avg_success_iteration * float(mj_model.opt.timestep)
                )
                frequencies[c_idx, h_idx] = result.control_frequency_hz

                state_store[c_idx][h_idx] = result.state_trajectories
                control_store[c_idx][h_idx] = result.control_trajectories
                trace_store[c_idx][h_idx] = result.trace_trajectories

        out_dir = self._save_results(
            horizons=horizons,
            success=success,
            success_time=success_time,
            frequencies=frequencies,
            state_store=state_store,
            control_store=control_store,
            trace_store=trace_store,
        )
        print(f"Saved benchmark outputs to {out_dir}")
        return out_dir

    def _save_results(
        self,
        *,
        horizons: np.ndarray,
        success: np.ndarray,
        success_time: np.ndarray,
        frequencies: np.ndarray,
        state_store: list,
        control_store: list,
        trace_store: list,
    ) -> Path:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        out_dir = (
            Path(ROOT)
            / "benchmark"
            / f"{self.task_name}_benchmark_{self.config.output_tag}_{timestamp}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savetxt(out_dir / "horizon_success_rate.csv", success, delimiter=",")
        np.savetxt(out_dir / "horizon_success_time.csv", success_time, delimiter=",")
        np.savetxt(out_dir / "horizon_frequency.csv", frequencies, delimiter=",")

        np.savez(
            out_dir / "summary.npz",
            horizons=horizons,
            success=success,
            success_time=success_time,
            frequencies=frequencies,
            controller_names=np.array([s.name for s in self.controller_specs], dtype=object),
        )

        np.savez(
            out_dir / "trajectories.npz",
            state_trajectories=np.array(state_store, dtype=object),
            control_trajectories=np.array(control_store, dtype=object),
            trace_trajectories=np.array(trace_store, dtype=object),
        )

        metadata = {
            "task_name": self.task_name,
            "controller_names": [s.name for s in self.controller_specs],
            "config": asdict(self.config),
            "horizons": horizons.tolist(),
        }
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self._plot_matrix(
            horizons,
            success,
            ylabel="Success Rate (%)",
            title=f"{self.task_name}: Success vs Horizon",
            out_path=out_dir / "horizon_success_rate.png",
        )
        self._plot_matrix(
            horizons,
            success_time,
            ylabel="Average Time-To-Success (s)",
            title=f"{self.task_name}: Time-To-Success vs Horizon",
            out_path=out_dir / "horizon_success_time.png",
        )
        self._plot_matrix(
            horizons,
            frequencies,
            ylabel="Control Frequency (Hz)",
            title=f"{self.task_name}: Control Frequency vs Horizon",
            out_path=out_dir / "horizon_frequency.png",
        )

        return out_dir

    def _plot_matrix(
        self,
        horizons: np.ndarray,
        values: np.ndarray,
        *,
        ylabel: str,
        title: str,
        out_path: Path,
    ) -> None:
        plt.figure()
        for idx, spec in enumerate(self.controller_specs):
            plt.plot(horizons, values[idx], label=spec.name)
        plt.title(title)
        plt.xlabel("Horizon (s)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

"""MPPI with staged rollout and KDE-based resampling."""

import numpy as np
from typing import Callable, Literal, Optional

from mppi_mjwarp.algs.mppi_base import MPPIBase
from mppi_mjwarp.tasks.task_base import Task


class MPPIStagedRollout(MPPIBase):
    """MPPI with staged rollout and KDE resampling.

    Splits the planning horizon into stages. After each stage, computes
    a KDE over the reached states and resamples trajectories inversely
    proportional to density to encourage state-space exploration.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        num_knots_per_stage: int = 4,
        kde_bandwidth: float = 1.0,
        state_weight: Optional[np.ndarray] = None,
        state_selection_function: Optional[Callable] = None,
        state_selection_fields: tuple = ("qpos",),
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__(
            task=task,
            num_samples=num_samples,
            noise_level=noise_level,
            temperature=temperature,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
            seed=seed,
        )
        self.num_knots_per_stage = num_knots_per_stage
        self.kde_bandwidth = kde_bandwidth
        self.state_selection_fields = tuple(state_selection_fields)

        if state_weight is not None:
            self.state_weight = np.asarray(state_weight, dtype=np.float32)

        if state_selection_function is not None:
            self.state_selection_function = state_selection_function

    def _get_costs(self, controls: np.ndarray, knots: np.ndarray) -> np.ndarray:
        return self._staged_rollout(controls, knots)

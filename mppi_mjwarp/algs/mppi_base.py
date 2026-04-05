"""Base MPPI controller using mujoco_warp for parallel rollouts.

Provides shared logic for all MPPI variants: state management, rollout,
softmax weighting, warm start, and spline-based control interpolation.
"""

import math

import mujoco
import mujoco_warp as mjwarp
import numpy as np
from typing import Callable, Literal, Optional

from mppi_mjwarp.tasks.task_base import Task, ALL_FIELDS
from mppi_mjwarp.utils.kde import gaussian_kde
from mppi_mjwarp.utils.spline import get_interp_func


def extract_data_np(d, fields=None):
    """Extract fields from mujoco_warp Data as numpy arrays.

    Args:
        d: mujoco_warp Data object.
        fields: Tuple of field names to extract. None = all fields.

    Returns:
        Dict mapping field names to numpy arrays, each (nworld, ...).
    """
    if fields is None:
        fields = ALL_FIELDS
    return {f: getattr(d, f).numpy() for f in fields}


class MPPIBase:
    """Base class for MPPI controllers using mujoco_warp.

    Subclasses override ``_get_costs`` to change the rollout strategy
    (e.g. staged rollout with KDE, memory-augmented terminal cost).
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        seed: int = 0,
    ) -> None:
        self.task = task
        self.num_samples = num_samples
        self.noise_level = noise_level
        self.temperature = temperature
        self.plan_horizon = plan_horizon
        self.num_knots = num_knots
        self.iterations = iterations
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.interp_func = get_interp_func(spline_type)
        self.dt = task.dt
        self.ctrl_steps = int(round(plan_horizon / self.dt))

        self.tk = np.linspace(0.0, plan_horizon, num_knots, dtype=np.float32)
        self._tq_relative = np.linspace(
            0.0, plan_horizon, self.ctrl_steps, dtype=np.float32
        )
        self.mean = np.zeros((num_knots, task.nu), dtype=np.float32)

        # Staged rollout defaults (overridden by subclasses that use it)
        self.num_knots_per_stage = None
        self.kde_bandwidth = None
        self.state_weight = np.ones((1,), dtype=np.float32)
        self.state_selection_function: Callable = lambda data_np: data_np["qpos"]
        self.state_selection_fields: tuple = ("qpos",)

        self.warp_data = mjwarp.make_data(task.mj_model, nworld=num_samples)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        initial_knots: Optional[np.ndarray] = None,
    ) -> None:
        """Reset controller state for a new trial."""
        if seed is None:
            seed = self.seed
        self.rng = np.random.default_rng(seed)
        self.tk = np.linspace(
            0.0, self.plan_horizon, self.num_knots, dtype=np.float32
        )
        if initial_knots is None:
            self.mean = np.zeros(
                (self.num_knots, self.task.nu), dtype=np.float32
            )
        else:
            knots = np.asarray(initial_knots, dtype=np.float32)
            expected = (self.num_knots, self.task.nu)
            if knots.shape != expected:
                raise ValueError(
                    f"initial_knots shape {knots.shape} != expected {expected}"
                )
            self.mean = knots.copy()

    def set_state_from_mj_data(self, mj_data: mujoco.MjData) -> None:
        """Broadcast a single MuJoCo state to all parallel worlds."""
        nw = self.num_samples
        self.warp_data.qpos.assign(
            np.tile(mj_data.qpos.astype(np.float32), (nw, 1))
        )
        self.warp_data.qvel.assign(
            np.tile(mj_data.qvel.astype(np.float32), (nw, 1))
        )
        self.warp_data.time.assign(
            np.full(nw, mj_data.time, dtype=np.float32)
        )
        if mj_data.mocap_pos.shape[0] > 0:
            self.warp_data.mocap_pos.assign(
                np.tile(mj_data.mocap_pos.astype(np.float32), (nw, 1, 1)).reshape(nw, -1, 3)
            )
            self.warp_data.mocap_quat.assign(
                np.tile(mj_data.mocap_quat.astype(np.float32), (nw, 1, 1)).reshape(nw, -1, 4)
            )

    def _save_state(self) -> dict:
        return {
            "qpos": self.warp_data.qpos.numpy().copy(),
            "qvel": self.warp_data.qvel.numpy().copy(),
            "time": self.warp_data.time.numpy().copy(),
        }

    def _restore_state(self, state: dict) -> None:
        self.warp_data.qpos.assign(state["qpos"])
        self.warp_data.qvel.assign(state["qvel"])
        self.warp_data.time.assign(state["time"])

    # ------------------------------------------------------------------
    # Warm start & action query (called by simulation runner)
    # ------------------------------------------------------------------

    def warm_start(self, current_time: float) -> None:
        """Advance knot times and re-evaluate the spline at the new times.

        This mirrors the Hydrax controller workflow: every optimize step
        shifts the spline support forward to the current simulation time
        before sampling around the previous mean.
        """
        new_tk = (
            np.linspace(0.0, self.plan_horizon, self.num_knots, dtype=np.float32)
            + current_time
        )
        new_mean = self.interp_func(new_tk, self.tk, self.mean[None, ...])[0]
        self.tk = new_tk
        self.mean = new_mean

    def get_action(self, t: float) -> np.ndarray:
        """Query the control spline at time *t*."""
        return self.interp_func(
            np.array([t], dtype=np.float32),
            self.tk,
            self.mean[None, ...],
        )[0, 0, :]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_knots(self):
        """Sample noisy knots and interpolate to a full control sequence.

        Returns:
            knots:    (num_samples, num_knots, nu)
            controls: (num_samples, ctrl_steps, nu)
        """
        noise = self.rng.standard_normal(
            (self.num_samples, self.num_knots, self.task.nu)
        ).astype(np.float32)
        knots = self.mean + self.noise_level * noise
        knots = np.clip(knots, self.task.u_min, self.task.u_max)

        tq = self._tq_relative + self.tk[0]
        controls = self.interp_func(tq, self.tk, knots)
        return knots, controls

    # ------------------------------------------------------------------
    # Rollout helpers
    # ------------------------------------------------------------------

    def _rollout(self, controls: np.ndarray) -> np.ndarray:
        """Forward-simulate all samples and collect costs.

        Args:
            controls: (num_samples, ctrl_steps, nu)

        Returns:
            costs: (num_samples, ctrl_steps + 1)  — running costs + terminal.
        """
        num_samples, ctrl_steps, _ = controls.shape
        costs = np.zeros((num_samples, ctrl_steps + 1), dtype=np.float32)

        rc_fields = self.task.running_cost_fields
        has_rc_state = len(rc_fields) > 0

        for t in range(ctrl_steps):
            ctrl_t = controls[:, t, :]
            self.warp_data.ctrl.assign(ctrl_t)
            mjwarp.step(self.task.model, self.warp_data)
            if has_rc_state:
                data_np = extract_data_np(self.warp_data, rc_fields)
            else:
                data_np = {}
            costs[:, t] = self.dt * self.task.running_cost(data_np, ctrl_t)

        costs[:, ctrl_steps] = self._compute_terminal_cost()
        return costs

    def _compute_terminal_cost(self) -> np.ndarray:
        """Compute terminal cost for all worlds.

        Overridden by MPPIMemoryContinuous to blend with memory heuristic.
        """
        data_np = extract_data_np(self.warp_data, self.task.terminal_cost_fields)
        return self.task.terminal_cost(data_np)

    def _staged_rollout(
        self, controls: np.ndarray, knots: np.ndarray
    ) -> np.ndarray:
        """Forward-simulate with KDE-based resampling at stage boundaries.

        Requires ``num_knots_per_stage`` and ``kde_bandwidth`` to be set
        (done by MPPIStagedRollout and MPPIMemoryContinuous).

        Args:
            controls: (num_samples, ctrl_steps, nu)
            knots:    (num_samples, num_knots, nu)

        Returns:
            costs: (num_samples, ctrl_steps + 1)
        """
        num_samples, ctrl_steps, _ = controls.shape
        costs = np.zeros((num_samples, ctrl_steps + 1), dtype=np.float32)

        num_stages = int(math.floor(self.num_knots / self.num_knots_per_stage))
        if num_stages <= 1:
            return self._rollout(controls)

        timesteps_per_stage = (
            int(math.floor(self.ctrl_steps / self.num_knots))
            * self.num_knots_per_stage
        )

        rc_fields = self.task.running_cost_fields
        has_rc_state = len(rc_fields) > 0

        for n in range(num_stages - 1):
            t_start = n * timesteps_per_stage
            t_end = (n + 1) * timesteps_per_stage

            # Partial rollout for this stage
            for t in range(t_start, t_end):
                ctrl_t = controls[:, t, :]
                self.warp_data.ctrl.assign(ctrl_t)
                mjwarp.step(self.task.model, self.warp_data)
                if has_rc_state:
                    data_np = extract_data_np(self.warp_data, rc_fields)
                else:
                    data_np = {}
                costs[:, t] = self.dt * self.task.running_cost(data_np, ctrl_t)

            # KDE resampling at stage boundary
            data_np = extract_data_np(self.warp_data, self.state_selection_fields)
            latest_state = np.asarray(
                self.state_selection_function(data_np), dtype=np.float32
            )
            weighted_state = latest_state * self.state_weight

            kde = gaussian_kde(weighted_state.T, bw=self.kde_bandwidth)
            p_x = kde.pdf(weighted_state.T)
            inv_px = 1.0 / (p_x + 1e-6)
            inv_px = inv_px / inv_px.sum()

            indices = self.rng.choice(num_samples, size=num_samples, p=inv_px)

            # Reorder everything by resampled indices
            controls = controls[indices]
            knots = knots[indices]
            costs = costs[indices]

            qpos_r = self.warp_data.qpos.numpy()[indices]
            qvel_r = self.warp_data.qvel.numpy()[indices]
            time_r = self.warp_data.time.numpy()[indices]
            self.warp_data.qpos.assign(qpos_r)
            self.warp_data.qvel.assign(qvel_r)
            self.warp_data.time.assign(time_r)

            # Resample remaining knots from mean + fresh noise
            k_start = (n + 1) * self.num_knots_per_stage
            remaining_mean = self.mean[k_start:, :]
            if remaining_mean.shape[0] > 0:
                remaining_noise = self.rng.standard_normal(
                    (num_samples, remaining_mean.shape[0], self.task.nu)
                ).astype(np.float32)
                new_partial = remaining_mean[None, :, :] + self.noise_level * remaining_noise
                new_partial = np.clip(new_partial, self.task.u_min, self.task.u_max)
                knots[:, k_start:, :] = new_partial

                controls = self.interp_func(
                    self._tq_relative + self.tk[0], self.tk, knots
                )

        # Final stage
        t_start = (num_stages - 1) * timesteps_per_stage
        for t in range(t_start, ctrl_steps):
            ctrl_t = controls[:, t, :]
            self.warp_data.ctrl.assign(ctrl_t)
            mjwarp.step(self.task.model, self.warp_data)
            if has_rc_state:
                data_np = extract_data_np(self.warp_data, rc_fields)
            else:
                data_np = {}
            costs[:, t] = self.dt * self.task.running_cost(data_np, ctrl_t)

        costs[:, ctrl_steps] = self._compute_terminal_cost()
        return costs

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def _update_weights(self, costs: np.ndarray, knots: np.ndarray) -> np.ndarray:
        """Compute softmax weights and update the mean.

        Returns:
            total_costs: (num_samples,)
        """
        total_costs = costs.sum(axis=1)
        shifted = -total_costs / self.temperature
        shifted -= shifted.max()
        weights = np.exp(shifted)
        weights /= weights.sum()
        self.mean = np.sum(weights[:, None, None] * knots, axis=0)
        return total_costs

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def optimize(self, mj_data: mujoco.MjData) -> np.ndarray:
        """Run one MPPI optimization step.

        Args:
            mj_data: Current simulation state.

        Returns:
            Updated mean control knots, shape (num_knots, nu).
        """
        self.warm_start(float(mj_data.time))
        self.set_state_from_mj_data(mj_data)
        init_state = self._save_state()

        for _ in range(self.iterations):
            self._restore_state(init_state)
            knots, controls = self._sample_knots()
            costs = self._get_costs(controls, knots)
            self._update_weights(costs, knots)

        return self.mean

    def _get_costs(
        self, controls: np.ndarray, knots: np.ndarray
    ) -> np.ndarray:
        """Override point for subclasses. Default: plain rollout."""
        return self._rollout(controls)

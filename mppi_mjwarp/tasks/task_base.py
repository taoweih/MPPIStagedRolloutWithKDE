"""Base task interface for mujoco_warp-based MPPI."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import mujoco
import mujoco_warp as mjwarp

# Package root for model paths
ROOT = str(Path(__file__).parent.parent.absolute())

# All extractable fields from mujoco_warp Data
ALL_FIELDS = ("xpos", "qpos", "qvel", "xquat", "sensordata", "site_xpos", "qfrc_constraint")


class Task(ABC):
    """Abstract task defining dynamics and cost functions for mujoco_warp.

    The task defines the discrete-time optimal control problem:

        min_u  phi(x_T) + sum_t  l(x_t, u_t)
        s.t.   x_{t+1} = f(x_t, u_t)

    where dynamics are defined by a MuJoCo model and costs by the task.

    Cost functions receive numpy arrays extracted from mujoco_warp Data,
    batched along the first axis (nworld).

    Subclasses should override running_cost_fields and terminal_cost_fields
    to declare which data fields they access. This enables selective GPU→CPU
    transfer, dramatically reducing overhead (e.g. control-only running costs
    skip all GPU extraction during rollout).
    """

    # Override in subclasses: which data fields running_cost / terminal_cost access.
    # Empty tuple = cost depends only on control (no GPU extraction needed).
    running_cost_fields: tuple = ALL_FIELDS
    terminal_cost_fields: tuple = ALL_FIELDS

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        trace_sites: Optional[Sequence[str]] = None,
        trace_bodies: Optional[Sequence[str]] = None,
    ) -> None:
        assert isinstance(mj_model, mujoco.MjModel)
        self.mj_model = mj_model

        # Put model onto GPU via mujoco_warp
        self.model = mjwarp.put_model(mj_model)

        # Actuator limits
        self.u_min = np.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 0],
            -np.inf,
        ).astype(np.float32)
        self.u_max = np.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 1],
            np.inf,
        ).astype(np.float32)

        # Simulation timestep
        self.dt = mj_model.opt.timestep

        # Convenience
        self.nu = mj_model.nu
        self.nq = mj_model.nq
        self.nv = mj_model.nv

        # Optional trace entities used by visualization and benchmark logging.
        self.trace_site_ids: list[int] = []
        self.trace_body_ids: list[int] = []

        for site_name in trace_sites or []:
            site_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )
            if site_id < 0:
                raise ValueError(f"Unknown trace site: {site_name}")
            self.trace_site_ids.append(site_id)

        for body_name in trace_bodies or []:
            body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id < 0:
                raise ValueError(f"Unknown trace body: {body_name}")
            self.trace_body_ids.append(body_id)

    @abstractmethod
    def running_cost(self, data_np: dict, control: np.ndarray) -> np.ndarray:
        """Compute running cost for all worlds.

        Args:
            data_np: Dict of numpy arrays extracted from mujoco_warp Data.
                     Keys: 'xpos' (nworld, nbody, 3), 'qpos' (nworld, nq),
                     'qvel' (nworld, nv), 'qfrc_constraint' (nworld, nv),
                     'xquat' (nworld, nbody, 4), 'sensordata' (nworld, nsensordata),
                     'site_xpos' (nworld, nsite, 3).
            control: Control actions, shape (nworld, nu).

        Returns:
            Running costs, shape (nworld,).
        """
        pass

    @abstractmethod
    def terminal_cost(self, data_np: dict) -> np.ndarray:
        """Compute terminal cost for all worlds.

        Args:
            data_np: Dict of numpy arrays from mujoco_warp Data.

        Returns:
            Terminal costs, shape (nworld,).
        """
        pass

    def success_function(self, data_np: dict, control: np.ndarray) -> np.ndarray:
        """Success metric (defaults to running_cost)."""
        return self.running_cost(data_np, control)

    def get_trace_positions(self, data_np: dict) -> np.ndarray:
        """Return trace points for each world, shape (nworld, ntrace, 3)."""
        nworld = None
        if data_np:
            first_val = next(iter(data_np.values()))
            nworld = int(first_val.shape[0])

        chunks = []

        if self.trace_site_ids:
            if "site_xpos" not in data_np:
                raise KeyError("site_xpos required for trace site visualization")
            chunks.append(data_np["site_xpos"][:, self.trace_site_ids, :])

        if self.trace_body_ids:
            if "xpos" not in data_np:
                raise KeyError("xpos required for trace body visualization")
            chunks.append(data_np["xpos"][:, self.trace_body_ids, :])

        if not chunks:
            if nworld is None:
                nworld = 1
            return np.zeros((nworld, 0, 3), dtype=np.float32)

        return np.concatenate(chunks, axis=1).astype(np.float32)

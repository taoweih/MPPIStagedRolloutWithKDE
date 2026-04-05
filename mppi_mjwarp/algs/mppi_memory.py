"""Memory-augmented MPPI with optional staged rollout."""

from __future__ import annotations

from typing import Callable, Literal, Optional, Sequence

import mujoco
import numpy as np

from mppi_mjwarp.algs.mppi_base import MPPIBase, extract_data_np
from mppi_mjwarp.algs.torch_memory import (
    MemoryPretrainConfig,
    TorchMemoryValueModel,
)
from mppi_mjwarp.tasks.task_base import Task


class MPPIMemoryContinuous(MPPIBase):
    """MPPI augmented with a learned value heuristic for terminal cost.

    Optionally combines with staged rollout (KDE resampling).  Memory
    pretraining and online updates are handled internally so benchmark
    scripts only need to configure callbacks and run trials.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        *,
        # Staged rollout params (used when use_staged_rollout=True)
        num_knots_per_stage: int = 4,
        kde_bandwidth: float = 1.0,
        state_weight: Optional[np.ndarray] = None,
        use_staged_rollout: bool = False,
        # Base params
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        seed: int = 0,
        # State selection
        state_selection_function: Optional[Callable[[dict], np.ndarray]] = None,
        state_selection_fields: Sequence[str] = ("qpos",),
        # Memory model params
        memory_state_dim: int = 2,
        memory_state_min: float | np.ndarray = -10.0,
        memory_state_max: float | np.ndarray = 10.0,
        memory_hidden_dim: int = 128,
        memory_num_layers: int = 2,
        memory_device: Optional[str] = None,
        memory_terminal_mix: float = 1.0,
        # Online learning params
        online_learning_rate: float = 1e-3,
        online_update_steps: int = 1,
        online_batch_size: int = 2048,
        online_anchor_samples: int = 4096,
        online_new_state_weight: float = 100.0,
        online_anchor_weight: float = 1.0,
        # Goal anchoring
        goal_state: Optional[np.ndarray] = None,
        goal_value: float = 0.0,
        goal_weight: float = 2000.0,
        # Auto pretrain
        auto_pretrain: bool = False,
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

        # Staged rollout configuration
        self.use_staged_rollout = use_staged_rollout
        self.num_knots_per_stage = num_knots_per_stage
        self.kde_bandwidth = kde_bandwidth
        self.state_selection_fields = tuple(state_selection_fields)

        if state_weight is not None:
            self.state_weight = np.asarray(state_weight, dtype=np.float32)
        if state_selection_function is not None:
            self.state_selection_function = state_selection_function

        # Build memory model
        state_min = np.asarray(memory_state_min, dtype=np.float32)
        state_max = np.asarray(memory_state_max, dtype=np.float32)
        if state_min.ndim == 0:
            state_min = np.full((memory_state_dim,), float(state_min), dtype=np.float32)
        if state_max.ndim == 0:
            state_max = np.full((memory_state_dim,), float(state_max), dtype=np.float32)

        self.memory_model = TorchMemoryValueModel(
            input_dim=memory_state_dim,
            state_min=state_min,
            state_max=state_max,
            hidden_dim=memory_hidden_dim,
            num_hidden_layers=memory_num_layers,
            device=memory_device,
            seed=seed,
        )

        self.memory_terminal_mix = float(np.clip(memory_terminal_mix, 0.0, 1.0))
        self.online_learning_rate = online_learning_rate
        self.online_update_steps = max(int(online_update_steps), 1)
        self.online_batch_size = max(int(online_batch_size), 1)
        self.online_anchor_samples = max(int(online_anchor_samples), 1)
        self.online_new_state_weight = float(online_new_state_weight)
        self.online_anchor_weight = float(online_anchor_weight)

        self.goal_state = (
            None if goal_state is None
            else np.asarray(goal_state, dtype=np.float32).reshape(1, -1)
        )
        self.goal_value = float(goal_value)
        self.goal_weight = float(goal_weight)

        self.auto_pretrain = auto_pretrain
        self.memory_ready = False
        self._pretrained_weights: Optional[np.ndarray] = None
        self._state_sampler: Optional[Callable] = None
        self._target_function: Optional[Callable] = None
        self._pretrain_config = MemoryPretrainConfig()

    # ------------------------------------------------------------------
    # Pretraining
    # ------------------------------------------------------------------

    def configure_pretraining(
        self,
        state_sampler: Callable[[np.random.Generator, int], np.ndarray],
        target_function: Callable[[np.ndarray], np.ndarray],
        config: Optional[MemoryPretrainConfig] = None,
    ) -> None:
        """Attach task-specific memory pretraining callbacks."""
        self._state_sampler = state_sampler
        self._target_function = target_function
        if config is not None:
            self._pretrain_config = config

    def pretrain_memory(self, force: bool = False, verbose: bool = True) -> bool:
        """Pretrain memory model. Returns True if memory is available."""
        if self.memory_ready and not force:
            return True
        if self._state_sampler is None or self._target_function is None:
            return False

        cfg = self._pretrain_config
        states = np.asarray(
            self._state_sampler(self.rng, cfg.sample_count), dtype=np.float32
        )
        targets = np.asarray(
            self._target_function(states), dtype=np.float32
        ).reshape(-1)

        if verbose:
            print(
                f"Pretraining memory with {states.shape[0]} samples, "
                f"{cfg.train_steps} SGD steps."
            )

        self.memory_model.fit(
            states, targets,
            steps=cfg.train_steps,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            one_sided=False,
            l2=1e-6,
            verbose=verbose,
            print_every=cfg.print_every,
        )
        self._pretrained_weights = self.memory_model.copy_weights()
        self.memory_ready = True
        return True

    def restore_pretrained_memory(self) -> bool:
        """Restore memory weights to the pretrained snapshot."""
        if self._pretrained_weights is None:
            return False
        self.memory_model.load_weights(self._pretrained_weights)
        self.memory_ready = True
        return True

    # ------------------------------------------------------------------
    # Reset (extends base to handle memory)
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        initial_knots: Optional[np.ndarray] = None,
        reset_memory_to_pretrained: bool = True,
    ) -> None:
        super().reset(seed=seed, initial_knots=initial_knots)
        if reset_memory_to_pretrained and self._pretrained_weights is not None:
            self.memory_model.load_weights(self._pretrained_weights)
            self.memory_ready = True

    # ------------------------------------------------------------------
    # Terminal cost override
    # ------------------------------------------------------------------

    def _compute_terminal_cost(self) -> np.ndarray:
        """Blend task terminal cost with learned heuristic."""
        task_data = extract_data_np(self.warp_data, self.task.terminal_cost_fields)
        base_terminal = self.task.terminal_cost(task_data)

        if not self.memory_ready:
            return base_terminal.astype(np.float32)

        mem_data = extract_data_np(self.warp_data, self.state_selection_fields)
        mem_state = np.asarray(
            self.state_selection_function(mem_data), dtype=np.float32
        )
        mem_terminal = self.memory_model.predict(mem_state)

        mix = self.memory_terminal_mix
        return ((1.0 - mix) * base_terminal + mix * mem_terminal).astype(np.float32)

    # ------------------------------------------------------------------
    # Cost override
    # ------------------------------------------------------------------

    def _get_costs(self, controls: np.ndarray, knots: np.ndarray) -> np.ndarray:
        if self.use_staged_rollout:
            return self._staged_rollout(controls, knots)
        return self._rollout(controls)

    # ------------------------------------------------------------------
    # Optimize override (adds online memory update)
    # ------------------------------------------------------------------

    def optimize(self, mj_data: mujoco.MjData) -> np.ndarray:
        if self.auto_pretrain and not self.memory_ready:
            self.pretrain_memory(force=False, verbose=False)

        current_state_vec = self._current_state_vector(mj_data)

        # Run base optimization with Hydrax-style warm start.
        self.warm_start(float(mj_data.time))
        self.set_state_from_mj_data(mj_data)
        init_state = self._save_state()

        best_cost = np.inf
        for _ in range(self.iterations):
            self._restore_state(init_state)
            knots, controls = self._sample_knots()
            costs = self._get_costs(controls, knots)
            total_costs = self._update_weights(costs, knots)
            best_cost = min(best_cost, float(np.min(total_costs)))

        # Online memory update
        self._update_memory_online(
            states=current_state_vec[None, :],
            values=np.array([best_cost], dtype=np.float32),
        )
        return self.mean

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def _current_state_vector(self, mj_data: mujoco.MjData) -> np.ndarray:
        """Extract the memory state vector from single-world mj_data."""
        data_np = {
            "xpos": np.asarray(mj_data.xpos, dtype=np.float32)[None, ...],
            "qpos": np.asarray(mj_data.qpos, dtype=np.float32)[None, ...],
            "qvel": np.asarray(mj_data.qvel, dtype=np.float32)[None, ...],
            "xquat": np.asarray(mj_data.xquat, dtype=np.float32)[None, ...],
            "sensordata": np.asarray(mj_data.sensordata, dtype=np.float32)[None, ...],
            "site_xpos": np.asarray(mj_data.site_xpos, dtype=np.float32)[None, ...],
            "qfrc_constraint": np.asarray(
                mj_data.qfrc_constraint, dtype=np.float32
            )[None, ...],
        }
        state = np.asarray(self.state_selection_function(data_np), dtype=np.float32)
        if state.ndim == 1:
            return state
        if state.ndim == 2 and state.shape[0] == 1:
            return state[0]
        return state.reshape(-1)

    def _update_memory_online(
        self, states: np.ndarray, values: np.ndarray
    ) -> None:
        """RTAA*-style online memory update with anchor regularisation."""
        states = np.asarray(states, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if states.ndim == 1:
            states = states[None, :]

        # Anchor points: preserve existing predictions elsewhere
        anchors = self.rng.uniform(
            self.memory_model.state_min,
            self.memory_model.state_max,
            size=(self.online_anchor_samples, self.memory_model.input_dim),
        ).astype(np.float32)
        anchor_targets = self.memory_model.predict(anchors)

        all_states = [states, anchors]
        all_targets = [values, anchor_targets]
        all_weights = [
            np.full((states.shape[0],), self.online_new_state_weight, dtype=np.float32),
            np.full((anchors.shape[0],), self.online_anchor_weight, dtype=np.float32),
        ]

        if self.goal_state is not None:
            all_states.append(self.goal_state)
            all_targets.append(np.array([self.goal_value], dtype=np.float32))
            all_weights.append(np.array([self.goal_weight], dtype=np.float32))

        self.memory_model.fit(
            np.concatenate(all_states, axis=0),
            np.concatenate(all_targets, axis=0),
            steps=self.online_update_steps,
            batch_size=min(
                self.online_batch_size,
                sum(s.shape[0] for s in all_states),
            ),
            learning_rate=self.online_learning_rate,
            sample_weights=np.concatenate(all_weights, axis=0),
            one_sided=True,
            l2=0.0,
            verbose=False,
        )
        self.memory_ready = True

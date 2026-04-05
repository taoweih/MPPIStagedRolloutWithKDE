"""PyTorch-backed heuristic model for memory-augmented MPPI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - handled at runtime when memory is used
    torch = None
    nn = None


@dataclass
class MemoryPretrainConfig:
    """Configuration for memory pretraining."""

    sample_count: int = 100000
    train_steps: int = 3000
    batch_size: int = 4096
    learning_rate: float = 1e-3
    print_every: int = 200


if nn is not None:
    class _MemoryMLP(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_hidden_layers: int,
        ) -> None:
            super().__init__()
            if num_hidden_layers < 1:
                raise ValueError("num_hidden_layers must be >= 1")

            layers: list[nn.Module] = []
            in_dim = input_dim
            for _ in range(num_hidden_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.SiLU())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
else:
    class _MemoryMLP:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PyTorch is required for MPPIMemoryContinuous. "
                "Install torch in the environment used for mppi_mjwarp."
            )


class TorchMemoryValueModel:
    """Continuous value model trained with PyTorch."""

    def __init__(
        self,
        input_dim: int,
        state_min: np.ndarray,
        state_max: np.ndarray,
        *,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        device: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for MPPIMemoryContinuous. "
                "Install torch in the environment used for mppi_mjwarp."
            )

        self.input_dim = int(input_dim)
        self.state_min = np.asarray(state_min, dtype=np.float32).reshape(self.input_dim)
        self.state_max = np.asarray(state_max, dtype=np.float32).reshape(self.input_dim)
        self.rng = np.random.default_rng(seed)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        torch.manual_seed(seed)
        self.model = _MemoryMLP(
            input_dim=self.input_dim,
            hidden_dim=int(hidden_dim),
            num_hidden_layers=int(num_hidden_layers),
        ).to(self.device)
        self.model.float()

    def copy_weights(self) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }

    def load_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        copied = {
            name: tensor.detach().clone().to(self.device)
            for name, tensor in state_dict.items()
        }
        self.model.load_state_dict(copied)

    def _normalize(self, states: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.state_max - self.state_min, 1e-6)
        x01 = (states - self.state_min[None, :]) / denom[None, :]
        return 2.0 * np.clip(x01, 0.0, 1.0) - 1.0

    def predict(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim == 1:
            states = states[None, :]

        norm_states = self._normalize(states)
        state_tensor = torch.from_numpy(norm_states).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(state_tensor).squeeze(-1)
        return pred.detach().cpu().numpy().astype(np.float32)

    def fit(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        *,
        steps: int,
        batch_size: int,
        learning_rate: float,
        sample_weights: Optional[np.ndarray] = None,
        one_sided: bool = False,
        l2: float = 0.0,
        verbose: bool = False,
        print_every: int = 200,
    ) -> float:
        states = np.asarray(states, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32).reshape(-1)
        if states.ndim == 1:
            states = states[None, :]

        n = states.shape[0]
        if sample_weights is None:
            sample_weights = np.ones((n,), dtype=np.float32)
        else:
            sample_weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)

        norm_states = self._normalize(states)
        state_tensor = torch.from_numpy(norm_states).to(self.device)
        target_tensor = torch.from_numpy(targets).to(self.device)
        weight_tensor = torch.from_numpy(sample_weights).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=l2,
        )

        self.model.train()
        last_loss = 0.0
        total_steps = max(int(steps), 1)
        for step in range(total_steps):
            if batch_size >= n:
                batch_idx = torch.arange(n, device=self.device)
            else:
                idx_np = self.rng.integers(0, n, size=int(batch_size))
                batch_idx = torch.as_tensor(
                    idx_np,
                    device=self.device,
                    dtype=torch.long,
                )

            xb = state_tensor.index_select(0, batch_idx)
            yb = target_tensor.index_select(0, batch_idx)
            wb = weight_tensor.index_select(0, batch_idx)

            pred = self.model(xb).squeeze(-1)
            if one_sided:
                residual = torch.clamp(yb - pred, min=0.0)
            else:
                residual = pred - yb

            loss = torch.mean(wb * residual.square())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            last_loss = float(loss.detach().cpu())
            if verbose and step % max(int(print_every), 1) == 0:
                print(f"  memory step {step:5d} | loss={last_loss:.6f}")

        return last_loss

import math
import torch

# ──────────────────────────────────────────────
class LowPassFilter:

    def __init__(self, cutoff_hz: float, dt: float, init_mode: str = "first"):
        assert cutoff_hz > 0, "cutoff_hz must be > 0"
        assert dt > 0,        "dt must be > 0"
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        self.alpha = dt / (tau + dt)
        self.prev: torch.Tensor | None = None
        self.init_mode = init_mode  # "first" | "zero"

    def reset(self):
        self.prev = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.prev is None:
            self.prev = torch.zeros_like(x) if self.init_mode == "zero" else x.clone()
            return self.prev

        self.prev = self.prev + self.alpha * (x - self.prev)
        return self.prev
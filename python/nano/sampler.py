from __future__ import annotations

import torch

class Sampler:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            return torch.argmax(logits, dim=-1)


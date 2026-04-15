from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RealData:
    """
    Thin wrapper around real samples (features + padding mask).
    This makes the "real data" role explicit and keeps it separate from the
    generator and discriminator implementations.
    """

    features: torch.Tensor
    padding_mask: torch.Tensor

    def slice_like(self, other: torch.Tensor) -> torch.Tensor:
        """
        Slice real features to match a (possibly smaller) other tensor
        along (B, T) dimensions.
        """
        b_size = min(self.features.size(0), other.size(0))
        t_size = min(self.features.size(1), other.size(1))
        return self.features[:b_size, :t_size]


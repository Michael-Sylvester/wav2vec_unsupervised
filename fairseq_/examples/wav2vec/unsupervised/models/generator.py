from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from fairseq.modules import TransposeLast


class Generator(nn.Module):
    """
    Generator network that produces token logits (fake samples) from input features.
    Extracted from `wav2vec_u.py` to separate concerns.
    """

    def __init__(self, input_dim: int, output_dim: int, cfg):
        super().__init__()

        self.cfg = cfg
        self.output_dim = output_dim
        self.stride = cfg.generator_stride
        self.dropout = nn.Dropout(cfg.generator_dropout)
        self.batch_norm = cfg.generator_batch_norm != 0
        self.residual = cfg.generator_residual

        padding = (
            cfg.generator_kernel // 2 if cfg.generator_pad < 0 else cfg.generator_pad
        )
        self.proj = nn.Sequential(
            TransposeLast(),
            nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=cfg.generator_kernel,
                stride=cfg.generator_stride,
                dilation=cfg.generator_dilation,
                padding=padding,
                bias=cfg.generator_bias,
            ),
            TransposeLast(),
        )

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(input_dim)
            self.bn.weight.data.fill_(cfg.generator_batch_norm)
        if self.residual:
            self.in_proj = nn.Linear(input_dim, input_dim)

    def forward(
        self,
        dense_x: torch.Tensor,
        tokens: Optional[torch.Tensor],
        dense_padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}

        if self.batch_norm:
            dense_x = self.bn_padded_data(dense_x, dense_padding_mask)
        if self.residual:
            inter_x = self.in_proj(self.dropout(dense_x))
            dense_x = dense_x + inter_x
            result["inter_x"] = inter_x

        dense_x = self.dropout(dense_x)

        dense_x = self.proj(dense_x)
        if self.stride > 1:
            dense_padding_mask = dense_padding_mask[:, :: self.stride]

        if dense_padding_mask.size(1) != dense_x.size(1):
            new_padding = dense_padding_mask.new_zeros(dense_x.shape[:-1])
            diff = new_padding.size(1) - dense_padding_mask.size(1)

            if diff > 0:
                new_padding[:, diff:] = dense_padding_mask
            else:
                assert diff < 0
                new_padding = dense_padding_mask[:, :diff]

            dense_padding_mask = new_padding

        token_x = None
        if tokens is not None:
            token_x = dense_x.new_zeros(tokens.numel(), self.output_dim)
            token_x.scatter_(1, tokens.view(-1, 1).long(), 1)
            token_x = token_x.view(tokens.shape + (self.output_dim,))

        result["dense_x"] = dense_x
        if token_x is not None:
            result["token_x"] = token_x
        result["dense_padding_mask"] = dense_padding_mask

        return result

    def bn_padded_data(self, feature: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        normed_feature = feature.clone()
        normed_feature[~padding_mask] = self.bn(
            feature[~padding_mask].unsqueeze(-1)
        ).squeeze(-1)
        return normed_feature


from __future__ import annotations

import torch
import torch.nn as nn

from fairseq.modules import SamePad


class Discriminator(nn.Module):
    """
    Discriminator network that scores samples as real/fake.
    Extracted from `wav2vec_u.py` to separate concerns.
    """

    def __init__(self, dim: int, cfg):
        super().__init__()

        inner_dim = cfg.discriminator_dim
        kernel = cfg.discriminator_kernel
        dilation = cfg.discriminator_dilation
        self.max_pool = cfg.discriminator_max_pool

        if cfg.discriminator_causal:
            padding = kernel - 1
        else:
            padding = kernel // 2

        def make_conv(in_d, out_d, k, p=0, has_dilation=True):
            conv = nn.Conv1d(
                in_d,
                out_d,
                kernel_size=k,
                padding=p,
                dilation=dilation if has_dilation else 1,
            )
            if cfg.discriminator_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            elif cfg.discriminator_weight_norm:
                conv = nn.utils.weight_norm(conv)
            return conv

        inner_net = [
            nn.Sequential(
                make_conv(inner_dim, inner_dim, kernel, padding),
                SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
                nn.Dropout(cfg.discriminator_dropout),
                nn.GELU(),
            )
            for _ in range(cfg.discriminator_depth - 1)
        ] + [
            make_conv(inner_dim, 1, kernel, padding, has_dilation=False),
            SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
        ]

        if cfg.discriminator_linear_emb:
            emb_net = [make_conv(dim, inner_dim, 1)]
        else:
            emb_net = [
                make_conv(dim, inner_dim, kernel, padding),
                SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
            ]

        if cfg.discriminator_act_after_linear:
            emb_net.append(nn.GELU())

        self.net = nn.Sequential(
            *emb_net,
            nn.Dropout(cfg.discriminator_dropout),
            *inner_net,
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        x = x.transpose(1, 2)  # BTC -> BCT
        x = self.net(x)
        x = x.transpose(1, 2)
        x_sz = x.size(1)
        if padding_mask is not None and padding_mask.any() and padding_mask.dim() > 1:
            padding_mask = padding_mask[:, : x.size(1)]
            x[padding_mask] = float("-inf") if self.max_pool else 0
            x_sz = x_sz - padding_mask.sum(dim=-1)
        x = x.squeeze(-1)
        if self.max_pool:
            x, _ = x.max(dim=-1)
        else:
            x = x.sum(dim=-1)
            x = x / x_sz
        return x


"""The identity network: a pooled text embedding in, parameter heads out.

Two small independent residual expert trunks — one owning body-shape
parameters, one owning face-shape parameters (with the resting-eyelid head
riding the face trunk). Deliberately tiny: identity generation is a
deterministic mapping, and the heavy lifting lives in the frozen text
encoder shared with the texture pipeline.

This module imports torch at import time; import it lazily (the top-level
package does) so that schema/assembly-only installs never need torch.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["IdentityNetwork", "ResidualBlock", "ExpertTrunk"]


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.dense_in = nn.Linear(hidden, hidden)
        self.dense_out = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dense_out(torch.nn.functional.silu(self.dense_in(self.norm(x))))


class ExpertTrunk(nn.Module):
    def __init__(self, input_dim: int, hidden: int, blocks: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden) for _ in range(blocks))
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.input_proj(self.input_norm(embedding)))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_norm(hidden)


class IdentityNetwork(nn.Module):
    """Body and face trunks with config-declared heads.

    `body` rides the body trunk and `face` rides the face trunk, always.
    Two optional heads cover the component generations: `eyelid`
    (resting-eyelid expression, face trunk — earlier components) and
    `proportions` (skeletal proportions, body trunk — components that own
    them). Head outputs are standardized values; the component's stored
    per-head mean/std de-standardize them
    (see :mod:`character_factory.identity`).
    """

    def __init__(self, input_dim: int, hidden: int, blocks: int,
                 body_size: int, face_size: int,
                 eyelid_size: int = 0, proportion_size: int = 0):
        super().__init__()
        self.body_trunk = ExpertTrunk(input_dim, hidden, blocks)
        self.face_trunk = ExpertTrunk(input_dim, hidden, blocks)
        self.body_head = nn.Linear(hidden, body_size)
        self.face_head = nn.Linear(hidden, face_size)
        self.eyelid_head = (
            nn.Linear(hidden, eyelid_size) if eyelid_size else None
        )
        self.proportion_head = (
            nn.Linear(hidden, proportion_size) if proportion_size else None
        )

    def forward(self, embedding: torch.Tensor) -> dict:
        body_hidden = self.body_trunk(embedding)
        face_hidden = self.face_trunk(embedding)
        out = {
            "body": self.body_head(body_hidden),
            "face": self.face_head(face_hidden),
        }
        if self.eyelid_head is not None:
            out["eyelid"] = self.eyelid_head(face_hidden)
        if self.proportion_head is not None:
            out["proportions"] = self.proportion_head(body_hidden)
        return out

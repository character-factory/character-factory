"""The identity network: a pooled text embedding in, parameter heads out.

Two model generations live here. The current one
(:class:`JointGenerativeIdentity`) is a single joint GENERATIVE model: a
semantic-center regressor plus a conditional rectified flow sampling a
residual around it, over one 51-value state (body identity, skeletal
proportions, face identity). The earlier generation
(:class:`IdentityNetwork`) is a deterministic dual-expert regressor, kept
so components that shipped with it keep loading. Both are deliberately
tiny: the heavy lifting lives in the frozen text encoder shared with the
texture pipeline.

This module imports torch at import time; import it lazily (the top-level
package does) so that schema/assembly-only installs never need torch.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "CenterNetwork",
    "CenterTrunk",
    "ExpertTrunk",
    "FlowBlock",
    "IdentityFlowNetwork",
    "IdentityNetwork",
    "JointGenerativeIdentity",
    "ResidualBlock",
]


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


# --------------------------------------------------------------------------
# the joint generative identity model (component generation 0.1.1+)
# --------------------------------------------------------------------------
#
# One artifact, one joint state: body identity, skeletal proportions, and
# face identity occupy declared slices of a single 51-value vector. Two
# cooperating networks live inside the one component: a small center
# regressor that predicts the semantic center of the distribution for a
# prompt, and a conditional rectified flow that samples a residual around
# that center. Identity generation is therefore GENERATIVE: the same prompt
# with different seeds yields different, equally valid identities, and the
# character document records whichever values were drawn.
#
# Every architectural choice below is pinned by the published weights — the
# layer layout, the modulation ordering, the time-embedding scale, and the
# sampling schedule are facts of the weights format, not preferences.


def _flow_time_embedding(timestep: torch.Tensor, width: int) -> torch.Tensor:
    """Sinusoidal embedding of continuous flow time in [0, 1]."""
    import math

    half = width // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timestep.device, dtype=timestep.dtype)
        / max(half - 1, 1)
    )
    angles = timestep[:, None] * frequencies[None, :] * 1_000.0
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if embedding.shape[-1] < width:
        embedding = torch.nn.functional.pad(
            embedding, (0, width - embedding.shape[-1])
        )
    return embedding


class FlowBlock(nn.Module):
    """AdaLN residual block: conditioning modulates, never joins the state."""

    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.modulation = nn.Linear(hidden, hidden * 3)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, state: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift, gate = self.modulation(conditioning).chunk(3, dim=-1)
        return state + gate * self.mlp(self.norm(state) * (1.0 + scale) + shift)


class IdentityFlowNetwork(nn.Module):
    """The conditional velocity field over the joint identity state."""

    def __init__(self, text_dim: int, output_dim: int, hidden: int, blocks: int):
        super().__init__()
        self.text_norm = nn.LayerNorm(text_dim)
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.time_projection = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.input_projection = nn.Linear(output_dim, hidden)
        self.blocks = nn.ModuleList(FlowBlock(hidden) for _ in range(blocks))
        self.output_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.output_projection = nn.Linear(hidden, output_dim)
        self.null_embedding = nn.Parameter(torch.zeros(text_dim))
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.hidden = hidden

    def forward(
        self, state: torch.Tensor, timestep: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        conditioning = self.text_projection(self.text_norm(embedding))
        conditioning = conditioning + self.time_projection(
            _flow_time_embedding(timestep, self.hidden)
        )
        hidden = self.input_projection(state)
        for block in self.blocks:
            hidden = block(hidden, conditioning)
        return self.output_projection(self.output_norm(hidden))

    @torch.no_grad()
    def sample(
        self, embedding: torch.Tensor, *, steps: int, guidance: float,
        temperature: float, generator: "torch.Generator",
    ) -> torch.Tensor:
        """Integrate the velocity field from noise (t=1) to identity (t=0).

        Euler steps on a uniform descending schedule, with classifier-free
        guidance evaluated as one batched call against the learned null
        embedding. Noise is drawn on the CPU from the caller's generator
        and moved to the model's device, so a given seed reproduces the
        same identity on every device.
        """
        batch = embedding.shape[0]
        state = (
            temperature
            * torch.randn(batch, self.output_dim, generator=generator)
        ).to(embedding.device)
        null = self.null_embedding.unsqueeze(0).expand(batch, -1)
        schedule = torch.linspace(1.0, 0.0, steps + 1, device=embedding.device)
        for index in range(steps):
            current = schedule[index]
            following = schedule[index + 1]
            time = current.expand(batch)
            velocity = self(
                torch.cat((state, state), dim=0),
                torch.cat((time, time), dim=0),
                torch.cat((embedding, null), dim=0),
            )
            conditional, unconditional = velocity.chunk(2, dim=0)
            guided = unconditional + guidance * (conditional - unconditional)
            state = state + (following - current) * guided
        return state


class CenterTrunk(nn.Module):
    """Residual trunk of the semantic-center regressor."""

    def __init__(self, input_dim: int, hidden: int, blocks: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(blocks)
        )
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.input(self.input_norm(embedding)))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output_norm(hidden)


class CenterNetwork(nn.Module):
    """The semantic-center half of the joint model: prompt → distribution
    center, in standardized space. Body and proportions share one trunk;
    face identity has its own."""

    def __init__(self, input_dim: int, hidden: int, blocks: int,
                 body_size: int, proportion_size: int, face_size: int):
        super().__init__()
        self.body_trunk = CenterTrunk(input_dim, hidden, blocks)
        self.face_trunk = CenterTrunk(input_dim, hidden, blocks)
        self.body_head = nn.Linear(hidden, body_size)
        self.proportion_head = nn.Linear(hidden, proportion_size)
        self.face_head = nn.Linear(hidden, face_size)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        body_hidden = self.body_trunk(embedding)
        face_hidden = self.face_trunk(embedding)
        return torch.cat(
            (
                self.body_head(body_hidden),
                self.proportion_head(body_hidden),
                self.face_head(face_hidden),
            ),
            dim=-1,
        )


class JointGenerativeIdentity(nn.Module):
    """The complete generative identity model: center + flow, one artifact.

    The flow learns the residual distribution around the center's
    prediction, so a sample is ``center(prompt) + flow_sample(prompt)``.
    Passing no generator collapses to the deterministic center — a
    diagnostic mode, not the product path: identity generation is
    generative, and the seed is part of the create contract.
    """

    stochastic = True

    def __init__(self, flow: IdentityFlowNetwork, center: CenterNetwork,
                 head_sizes: list, sampling: dict):
        super().__init__()
        self.flow = flow
        self.center = center
        self.head_sizes = tuple((name, int(size)) for name, size in head_sizes)
        self.steps = int(sampling["steps"])
        self.guidance = float(sampling["guidance"])
        self.temperature = float(sampling["temperature"])

    def forward(
        self, embedding: torch.Tensor,
        generator: "torch.Generator | None" = None,
    ) -> dict[str, torch.Tensor]:
        center = self.center(embedding)
        if generator is None:
            standardized = center
        else:
            standardized = center + self.flow.sample(
                embedding, steps=self.steps, guidance=self.guidance,
                temperature=self.temperature, generator=generator,
            )
        output = {}
        offset = 0
        for name, size in self.head_sizes:
            output[name] = standardized[:, offset:offset + size]
            offset += size
        return output

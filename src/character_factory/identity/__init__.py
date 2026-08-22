"""Identity generation: raw text description → MHR body parameters.

Deterministic by contract (ARCHITECTURE.md §1): no seed anywhere in this
module. The same description, component version, and device produce the
same 45 identity coefficients and 72 resting-expression values.

The `identity` registry component is a directory of two files:

- ``config.json`` — the embedding recipe, architecture hyperparameters, and
  the owned-position index maps (which identity/expression positions each
  head writes). Everything index-shaped is component data, never code.
- ``weights.safetensors`` — the network's state dict plus per-head
  de-standardization statistics (``stats.<head>.mean`` / ``stats.<head>.std``).

Import note: this module needs torch (and, for the built-in embedder,
transformers) — install extra ``[generation]``. The top-level package
imports it lazily so schema/assembly-only installs never touch it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Protocol

__all__ = [
    "IdentityComponent",
    "IdentityGenerator",
    "IdentityResult",
    "TextEmbedder",
]

COMPONENT_FORMAT = "character-factory/identity-component"


class IdentityResult:
    """One generated identity: coefficients, resting expression, and named
    skeletal proportions ({} when the component does not own them)."""

    def __init__(self, identity: list, resting_expression: list,
                 proportions: dict):
        self.identity = identity
        self.resting_expression = resting_expression
        self.proportions = proportions


class TextEmbedder(Protocol):
    """Anything that turns one prompt into a pooled embedding vector."""

    def __call__(self, prompt: str) -> "torch.Tensor":  # noqa: F821 (lazy torch)
        ...


class IdentityComponent:
    """A loaded identity component: config + network + statistics."""

    def __init__(self, config: dict, model, stats: dict):
        self.config = config
        self.model = model
        self.stats = stats  # head -> {"mean": tensor, "std": tensor}

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> "IdentityComponent":
        import torch
        from safetensors.torch import load_file

        from character_factory.identity.model import IdentityNetwork

        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config.get("format") != COMPONENT_FORMAT:
            raise ValueError(
                f"{directory} is not an identity component "
                f"(format={config.get('format')!r})"
            )
        arch = config["architecture"]
        heads = config["heads"]
        model = IdentityNetwork(
            input_dim=config["embedding"]["dimensions"],
            hidden=arch["hidden"],
            blocks=arch["blocks"],
            body_size=len(heads["body"]["identity_indices"]),
            face_size=len(heads["face"]["identity_indices"]),
            eyelid_size=(
                len(heads["eyelid"]["expression_indices"])
                if "eyelid" in heads else 0
            ),
            proportion_size=(
                len(heads["proportions"]["parameters"])
                if "proportions" in heads else 0
            ),
        )
        tensors = load_file(str(directory / "weights.safetensors"), device=device)
        stats = {}
        state = {}
        for key, tensor in tensors.items():
            if key.startswith("stats."):
                _, head, kind = key.split(".")
                stats.setdefault(head, {})[kind] = tensor.float()
            else:
                state[key] = tensor
        model.load_state_dict(state)
        model.to(device).eval()
        for head in heads:
            if head not in stats or set(stats[head]) != {"mean", "std"}:
                raise ValueError(f"weights.safetensors is missing stats for {head!r}")
        with torch.no_grad():
            return cls(config, model, stats)


class IdentityGenerator:
    """Text → (identity[45], resting_expression[72]).

    The embedder is injectable; :meth:`with_base_model` builds the standard
    one (the base model's text encoder, masked-mean pooled and L2-normalized)
    while tests and alternative front-ends can supply their own.
    """

    def __init__(self, component: IdentityComponent, embedder: TextEmbedder):
        self.component = component
        self.embedder = embedder

    @classmethod
    def with_base_model(
        cls, component: IdentityComponent, base_model_dir: str | Path,
        device: str = "cpu",
    ) -> "IdentityGenerator":
        embedder = _transformers_embedder(
            Path(base_model_dir), component.config["embedding"], device
        )
        return cls(component, embedder)

    def generate(self, prompt: str) -> "IdentityResult":
        return self.generate_from_embedding(self.embedder(prompt))

    def generate_from_embedding(self, embedding) -> "IdentityResult":
        import torch

        config = self.component.config
        stats = self.component.stats
        with torch.no_grad():
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            raw = self.component.model(embedding.float())
            values = {
                head: (z[0] * stats[head]["std"] + stats[head]["mean"]).cpu()
                for head, z in raw.items()
            }

        heads = config["heads"]
        base = config.get("base_identity")
        identity = list(map(float, base)) if base else [0.0] * config["identity_size"]
        for index, value in zip(heads["body"]["identity_indices"], values["body"]):
            identity[index] = float(value)
        for index, value in zip(heads["face"]["identity_indices"], values["face"]):
            identity[index] = float(value)
        expression = [0.0] * config["expression_size"]
        if "eyelid" in heads:
            for index, value in zip(
                heads["eyelid"]["expression_indices"], values["eyelid"]
            ):
                expression[index] = float(value)
        proportions: dict[str, float] = {}
        if "proportions" in heads:
            spec = heads["proportions"]
            bound = float(spec.get("bound", 0.0)) or None
            for name, value in zip(spec["parameters"], values["proportions"]):
                v = float(value)
                if bound is not None:
                    v = max(-bound, min(bound, v))
                proportions[name] = v
        return IdentityResult(identity, expression, proportions)


def _transformers_embedder(
    base_model_dir: Path, embedding_config: dict, device: str
) -> Callable:
    """The standard embedder: the base model's frozen text encoder, masked-mean
    pooled over real tokens and L2-normalized. Loaded once, then reused —
    and releasable by dropping the returned closure."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir / "tokenizer"))
    encoder = (
        AutoModel.from_pretrained(str(base_model_dir / "text_encoder"), dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    max_tokens = embedding_config["max_tokens"]

    @torch.no_grad()
    def embed(prompt: str) -> torch.Tensor:
        tokens = tokenizer(
            [prompt], padding=True, truncation=True, max_length=max_tokens,
            return_tensors="pt",
        ).to(device)
        hidden = encoder(**tokens).last_hidden_state.float()
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        return torch.nn.functional.normalize(pooled, dim=-1)[0]

    return embed

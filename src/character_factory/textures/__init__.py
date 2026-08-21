"""Texture baking: seeded diffusion per texture slot ([generation] extra).

Component-driven end to end: the base model, per-slot adapters, prompt
templates, and sampler defaults all come from the registry; a recipe's
explicit seed (and optional overrides) come from the character file. The
pipeline object is injectable so the orchestration is testable without a
GPU; the default factory loads the base model once and hot-swaps adapters
between slots.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from character_factory.registry import ComponentEntry, Registry
from character_factory.schema import Character

__all__ = ["BakeResult", "TextureBaker", "bake"]


@dataclass
class BakeResult:
    character: Character          # with the assets block filled in
    assets_dir: Path
    baked_slots: list[str]


class TextureBaker:
    """Runs one adapter per slot against a shared base pipeline."""

    def __init__(self, registry: Registry, device: str = "cuda",
                 pipeline_factory=None):
        self.registry = registry
        self.device = device
        self._pipeline_factory = pipeline_factory or _default_pipeline_factory
        self._pipeline = None

    def _pipeline_for(self, entry: ComponentEntry):
        if self._pipeline is None:
            base_ref = entry.document.get("requires", {}).get("base_model")
            base_dir = self.registry.ensure(base_ref)
            self._pipeline = self._pipeline_factory(base_dir, self.device)
        return self._pipeline

    def close(self) -> None:
        """Drop the pipeline and hand its VRAM back to the driver.

        Long-lived processes (the server) run other GPU stages between
        bakes — most notably the interpreter model. Torch's allocator
        would otherwise keep the pipeline's memory reserved, and an
        over-subscribed card degrades to shared-memory paging (silently,
        on WSL2) instead of failing loudly.
        """
        import gc

        self._pipeline = None
        # Reference cycles keep the weights alive past the del; collect
        # before returning the cache or the VRAM stays resident.
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def bake_slot(self, slot: str, recipe: dict, out_path: Path) -> dict:
        """Generate one slot's image; returns its SPEC.md §8 asset descriptor."""
        entry = self.registry.get(recipe["component"], recipe["component_version"])
        adapter_dir = self.registry.ensure(entry.ref)
        pipeline = self._pipeline_for(entry)

        inference = {**entry.inference, **recipe.get("overrides", {})}
        template = inference.get("prompt_template", "{prompt}")
        resolution = inference.get("resolution", 1024)
        image = pipeline.generate(
            prompt=template.format(prompt=recipe["prompt"]),
            seed=recipe["seed"],
            steps=inference.get("steps", 20),
            guidance=inference.get("guidance", 4.0),
            resolution=resolution,
            adapter_dir=adapter_dir,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="PNG")
        data = out_path.read_bytes()
        return {
            "sha256": hashlib.sha256(data).hexdigest(),
            "media_type": "image/png",
            "width": image.width,
            "height": image.height,
        }


def bake(
    character: Character,
    out_dir: str | Path,
    *,
    registry: Registry | None = None,
    device: str = "cuda",
    pipeline_factory=None,
) -> BakeResult:
    """Bake every texture slot in the character; write `<slot>.png` files and
    return the character with its `assets` block pinned to what was written."""
    out_dir = Path(out_dir)
    registry = registry or Registry.default()
    baker = TextureBaker(registry, device=device, pipeline_factory=pipeline_factory)

    # v0.1 bakes each slot's albedo map; the flat asset descriptor is the
    # albedo shorthand (SPEC.md §5.2, §8). Albedo files are named <slot>.png;
    # future secondary maps will be <slot>.<map>.png.
    descriptors: dict[str, dict] = {}
    try:
        for slot, maps in sorted(character.texture_maps().items()):
            descriptors[slot] = baker.bake_slot(
                slot, maps["albedo"], out_dir / f"{slot}.png"
            )
    finally:
        baker.close()

    document = character.to_document()
    document["assets"] = descriptors
    return BakeResult(
        character=Character.from_document(document),
        assets_dir=out_dir,
        baked_slots=sorted(descriptors),
    )


# Weight quantization modes for the texture pipeline. Full precision is
# the default; the quantized modes trade a little fidelity for a much
# smaller resident model, which is what lets the bake fit smaller cards.
QUANTIZATION_MODES = ("nf4", "int8")
ENV_QUANTIZATION = "CHARACTER_FACTORY_TEXTURE_QUANTIZATION"


def configured_quantization() -> str | None:
    """The configured texture quantization mode, or None for full
    precision. Environment first, then the `textures.quantization` key of
    the cache config (the same file the interpreter and registry read) —
    configuration, never code."""
    import json
    import os

    from character_factory.registry.store import cache_dir

    mode = os.environ.get(ENV_QUANTIZATION)
    if not mode:
        path = cache_dir() / "config.json"
        if path.is_file():
            document = json.loads(path.read_text(encoding="utf-8"))
            section = document.get("textures", {}) if isinstance(document, dict) else {}
            mode = section.get("quantization")
    if not mode:
        return None
    if mode not in QUANTIZATION_MODES:
        raise ValueError(
            f"unknown texture quantization {mode!r}; "
            f"expected one of {', '.join(QUANTIZATION_MODES)}"
        )
    return mode


def _quantization_config(mode: str):
    """The diffusers pipeline-level quantization config for a mode. The
    transformer and text encoder carry nearly all of the weight memory;
    the VAE stays full precision (it is small and decode-critical)."""
    import torch
    from diffusers.quantizers import PipelineQuantizationConfig

    if mode == "nf4":
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["transformer", "text_encoder"],
        )
    return PipelineQuantizationConfig(
        quant_backend="bitsandbytes_8bit",
        quant_kwargs={"load_in_8bit": True},
        components_to_quantize=["transformer", "text_encoder"],
    )


class _DiffusersPipeline:
    """The real backend: the base model via diffusers, adapters hot-swapped.

    Wrapped behind the same small interface the tests fake, so orchestration
    and GPU specifics stay separable.
    """

    def __init__(self, base_dir: Path, device: str,
                 quantization: str | None = None):
        import torch
        from diffusers import DiffusionPipeline

        kwargs = {"torch_dtype": torch.bfloat16}
        if quantization is not None:
            kwargs["quantization_config"] = _quantization_config(quantization)
        # The base component's model_index.json declares the pipeline class;
        # resolving it from the distribution keeps this code base-model-
        # agnostic (a registry data change, not a code change).
        self.pipeline = DiffusionPipeline.from_pretrained(
            str(base_dir), **kwargs
        ).to(device)
        if quantization is not None and hasattr(self.pipeline, "vae"):
            # Quantized models report float32, so the pipeline prepares
            # float32 latents; the (small) VAE runs float32 to match.
            self.pipeline.vae.to(torch.float32)
        self.device = device
        self._adapter: Path | None = None

    def generate(self, *, prompt, seed, steps, guidance, resolution, adapter_dir):
        import torch

        if self._adapter != adapter_dir:
            self.pipeline.unload_lora_weights()
            # Component convention: every component's weight artifact is
            # weights.safetensors.
            self.pipeline.load_lora_weights(
                str(adapter_dir), weight_name="weights.safetensors"
            )
            self._adapter = adapter_dir
        generator = torch.Generator(self.device).manual_seed(seed)
        return self.pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=resolution,
            height=resolution,
            generator=generator,
        ).images[0]


def _default_pipeline_factory(base_dir: Path, device: str):
    return _DiffusersPipeline(base_dir, device,
                              quantization=configured_quantization())

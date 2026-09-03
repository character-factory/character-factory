"""Texture baking: seeded diffusion per texture slot ([generation] extra).

Component-driven end to end: the base model, per-slot adapters, prompt
templates, and sampler defaults all come from the registry; a recipe's
explicit seed (and optional overrides) come from the character file. The
pipeline object is injectable so the orchestration is testable without a
GPU; the default factory loads the base model once and hot-swaps adapters
between slots.

A bake runs in two GPU phases so the base model's text encoder and its
transformer are never resident together: every slot's prompt is encoded
first, the encoder is released, and only then does the transformer load
and denoise from the stored conditioning. The bake's peak memory is the
larger phase rather than their sum.
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
    """Runs one adapter per slot against a shared base pipeline.

    `turbo` swaps the adapters' declared base model for the fast variant
    the base's registry entry names (`turbo_variant` — a distilled
    sibling), and adopts that variant's declared sampling regime (its
    `inference` steps/guidance) in place of each adapter's. The adapters
    themselves are unchanged; explicit recipe overrides still win.
    """

    def __init__(self, registry: Registry, device: str = "cuda",
                 pipeline_factory=None, turbo: bool = False):
        self.registry = registry
        self.device = device
        self.turbo = turbo
        self._pipeline_factory = pipeline_factory or _default_pipeline_factory
        self._pipeline = None
        self._turbo_overrides: dict = {}

    def _pipeline_for(self, entry: ComponentEntry):
        if self._pipeline is None:
            base_ref = entry.document.get("requires", {}).get("base_model")
            if self.turbo:
                base_entry = self.registry.get(base_ref)
                turbo_ref = base_entry.document.get("turbo_variant")
                if not turbo_ref:
                    raise ValueError(
                        f"base model {base_ref!r} declares no turbo variant"
                    )
                turbo_entry = self.registry.get(turbo_ref)
                inference = turbo_entry.document.get("inference", {}) or {}
                self._turbo_overrides = {
                    key: inference[key]
                    for key in ("steps", "guidance") if key in inference
                }
                base_dir = self.registry.ensure(turbo_entry.ref)
            else:
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
        self._pipeline = None
        # Reference cycles keep the weights alive past the del; collect
        # before returning the cache or the VRAM stays resident.
        _release_device()

    def _inference(self, entry: ComponentEntry, recipe: dict) -> dict:
        return {
            **entry.inference,
            **self._turbo_overrides,
            **recipe.get("overrides", {}),
        }

    def slot_prompt(self, recipe: dict) -> str:
        """The caption a recipe conditions on: the component's declared
        template filled from the recipe (SPEC.md §5.2)."""
        entry = self.registry.get(recipe["component"], recipe["component_version"])
        inference = self._inference(entry, recipe)
        template = inference.get("prompt_template", "{prompt}")
        fields = {"prompt": recipe["prompt"]}
        if "{shaft_clause}" in template:
            # Style-dependent conditioning: the component's foot chart maps
            # the recipe prompt to a style, whose shaft clause (declared in
            # the registry entry) completes the caption.
            from character_factory.assembly.footwear import FootChart, shaft_clause

            chart = FootChart.load(self.registry.ensure(entry.ref))
            fields["shaft_clause"] = shaft_clause(
                chart.style_for_prompt(recipe["prompt"]), inference
            )
        return template.format(**fields)

    def encode(self, recipes: list[dict]) -> None:
        """Encode every recipe's caption up front, on a pipeline that
        supports it (the default one does; a fake may not). The encoder
        runs alone, then leaves the device before the transformer loads."""
        if not recipes:
            return
        entry = self.registry.get(
            recipes[0]["component"], recipes[0]["component_version"]
        )
        pipeline = self._pipeline_for(entry)
        encode = getattr(pipeline, "encode", None)
        if encode is not None:
            encode([self.slot_prompt(recipe) for recipe in recipes])

    def bake_slot(self, slot: str, recipe: dict, out_path: Path) -> dict:
        """Generate one slot's image; returns its SPEC.md §8 asset descriptor."""
        entry = self.registry.get(recipe["component"], recipe["component_version"])
        adapter_dir = self.registry.ensure(entry.ref)
        pipeline = self._pipeline_for(entry)
        inference = self._inference(entry, recipe)
        resolution = inference.get("resolution", 1024)
        image = pipeline.generate(
            prompt=self.slot_prompt(recipe),
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
    turbo: bool = False,
) -> BakeResult:
    """Bake every texture slot in the character; write `<slot>.png` files and
    return the character with its `assets` block pinned to what was written.

    `turbo` is a bake-time speed/quality trade (like quantization): the
    fast distilled base variant with its own sampling regime. It is not
    recorded in the character document — recipes stay the recipes."""
    out_dir = Path(out_dir)
    registry = registry or Registry.default()
    if pipeline_factory is None:
        # Fail in seconds with a named cause (missing dependency, CPU-only
        # torch, dead or too-old driver) before any pipeline loads. An
        # injected factory owns its own stack and is exempt (tests).
        from character_factory.preflight import require_generation_stack

        require_generation_stack(device)
    baker = TextureBaker(registry, device=device,
                         pipeline_factory=pipeline_factory, turbo=turbo)

    # v0.1 bakes each slot's albedo map; the flat asset descriptor is the
    # albedo shorthand (SPEC.md §5.2, §8). Albedo files are named <slot>.png;
    # future secondary maps will be <slot>.<map>.png.
    descriptors: dict[str, dict] = {}
    recipes = {
        slot: maps["albedo"]
        for slot, maps in sorted(character.texture_maps().items())
    }
    try:
        # All captions first, then all images: the text encoder and the
        # transformer take turns on the device instead of sharing it.
        baker.encode(list(recipes.values()))
        for slot, recipe in recipes.items():
            descriptors[slot] = baker.bake_slot(
                slot, recipe, out_dir / f"{slot}.png"
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
    and GPU specifics stay separable. The base model loads in two halves,
    never together: the text encoder (with its tokenizer) to turn captions
    into conditioning, then the transformer, VAE and scheduler to denoise
    from it. Captions encoded through `encode` are held on the device
    (a few megabytes each); `generate` encodes any caption it has not
    seen — correct either way, only the peak memory differs.
    """

    # Conditioning for an empty caption, which classifier-free guidance
    # pairs with every prompt.
    _NEGATIVE = ""

    def __init__(self, base_dir: Path, device: str,
                 quantization: str | None = None):
        self.base_dir = base_dir
        self.device = device
        self.quantization = quantization
        self.pipeline = None       # the denoising half, loaded on first use
        self._conditioning: dict[str, object] = {}   # caption → embeds
        self._adapter: Path | None = None

    def _load(self, **omit):
        import torch
        from diffusers import DiffusionPipeline

        kwargs = {"torch_dtype": torch.bfloat16, **omit}
        if self.quantization is not None:
            kwargs["quantization_config"] = _quantization_config(self.quantization)
        # The base component's model_index.json declares the pipeline class;
        # resolving it from the distribution keeps this code base-model-
        # agnostic (a registry data change, not a code change). Passing
        # None for a component leaves it unloaded.
        return DiffusionPipeline.from_pretrained(
            str(self.base_dir), **kwargs
        ).to(self.device)

    def encode(self, prompts: list[str]) -> None:
        """Encode captions (and the empty caption) with only the text
        encoder resident, then release it."""
        import torch

        wanted = [p for p in [*prompts, self._NEGATIVE] if p not in self._conditioning]
        if not wanted:
            return
        encoder = self._load(transformer=None, vae=None)
        try:
            with torch.no_grad():
                for prompt in wanted:
                    # The pipeline's own encode_prompt, with its own
                    # defaults — the same conditioning a plain prompt
                    # call would compute internally.
                    embeds, *_ = encoder.encode_prompt(
                        prompt=prompt, device=self.device
                    )
                    self._conditioning[prompt] = embeds
        finally:
            del encoder
            _release_device()

    def _denoiser(self):
        if self.pipeline is None:
            import torch

            self.pipeline = self._load(text_encoder=None, tokenizer=None)
            if self.quantization is not None and hasattr(self.pipeline, "vae"):
                # Quantized models report float32, so the pipeline prepares
                # float32 latents; the (small) VAE runs float32 to match.
                self.pipeline.vae.to(torch.float32)
        return self.pipeline

    def generate(self, *, prompt, seed, steps, guidance, resolution, adapter_dir):
        import torch

        if prompt not in self._conditioning:
            self.encode([prompt])
        pipeline = self._denoiser()
        if self._adapter != adapter_dir:
            pipeline.unload_lora_weights()
            # Component convention: every component's weight artifact is
            # weights.safetensors.
            pipeline.load_lora_weights(
                str(adapter_dir), weight_name="weights.safetensors"
            )
            self._adapter = adapter_dir
        generator = torch.Generator(self.device).manual_seed(seed)
        return pipeline(
            prompt_embeds=self._conditioning[prompt],
            negative_prompt_embeds=self._conditioning[self._NEGATIVE],
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=resolution,
            height=resolution,
            generator=generator,
        ).images[0]


def _release_device() -> None:
    """Collect dropped modules and hand their VRAM back to the driver."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _default_pipeline_factory(base_dir: Path, device: str):
    return _DiffusersPipeline(base_dir, device,
                              quantization=configured_quantization())

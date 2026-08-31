"""Validation of character documents against SPEC.md v0.1.

Two modes (SPEC.md §2, §10):

- **default** — unknown *optional* fields produce warnings, so documents from
  a newer schema minor version remain readable;
- **strict** — unknown fields are errors.

In both modes an unrecognized `body.rig` or `body.topology` value is a hard
error: those change what the document builds, not just what it records.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from character_factory.schema import vocab

__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "hair_block_errors",
    "validate_document",
]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"
)
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)$")


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"{self.path}: {self.message}"


@dataclass
class ValidationReport:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class _Checker:
    def __init__(self, strict: bool):
        self.strict = strict
        self.report = ValidationReport()

    # -- issue plumbing ----------------------------------------------------

    def error(self, path: str, message: str) -> None:
        self.report.errors.append(ValidationIssue(path, message))

    def unknown(self, path: str, message: str) -> None:
        """Unknown optional content: warning by default, error in strict."""
        issue = ValidationIssue(path, message)
        (self.report.errors if self.strict else self.report.warnings).append(issue)

    def check_keys(self, obj: dict, allowed: set[str], path: str) -> None:
        for key in obj:
            if key not in allowed:
                self.unknown(f"{path}.{key}", "unknown field")

    # -- scalar helpers ----------------------------------------------------

    def expect_str(self, obj: dict, key: str, path: str, required: bool = True) -> str | None:
        if key not in obj:
            if required:
                self.error(f"{path}.{key}", "required field is missing")
            return None
        value = obj[key]
        if not isinstance(value, str):
            self.error(f"{path}.{key}", "must be a string")
            return None
        return value

    def expect_number(self, value: object, path: str) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.error(path, "must be a number")
            return False
        if isinstance(value, float) and not math.isfinite(value):
            self.error(path, "must be finite (NaN and infinities are invalid)")
            return False
        return True

    def expect_seed(self, obj: dict, path: str) -> None:
        if "seed" not in obj:
            self.error(f"{path}.seed", "required field is missing")
            return
        seed = obj["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            self.error(f"{path}.seed", "must be an integer")
        elif not vocab.SEED_MIN <= seed <= vocab.SEED_MAX:
            self.error(f"{path}.seed", f"must be in [{vocab.SEED_MIN}, {vocab.SEED_MAX}]")

    def expect_enum(self, obj: dict, key: str, values: frozenset[str], path: str,
                    required: bool = True) -> None:
        raw = self.expect_str(obj, key, path, required=required)
        if raw is not None and raw not in values:
            self.error(f"{path}.{key}", f"must be one of: {', '.join(sorted(values))}")

    def expect_float_array(self, obj: dict, key: str, length: int, path: str) -> None:
        if key not in obj:
            self.error(f"{path}.{key}", "required field is missing")
            return
        value = obj[key]
        if not isinstance(value, list):
            self.error(f"{path}.{key}", "must be an array")
            return
        if len(value) != length:
            self.error(f"{path}.{key}", f"must have exactly {length} entries, got {len(value)}")
            return
        for i, item in enumerate(value):
            self.expect_number(item, f"{path}.{key}[{i}]")

    # -- blocks --------------------------------------------------------------

    def check_body(self, body: object) -> None:
        if not isinstance(body, dict):
            self.error("body", "must be an object")
            return
        self.check_keys(
            body,
            {"rig", "topology", "identity", "proportions", "resting_expression"},
            "body",
        )
        # rig/topology: unrecognized values are hard errors in every mode.
        rig = self.expect_str(body, "rig", "body")
        if rig is not None and rig not in vocab.RIGS:
            self.error("body.rig", f"unrecognized rig {rig!r}: this document requires a "
                                    "newer schema version than this implementation supports")
        topology = self.expect_str(body, "topology", "body")
        if topology is not None and topology not in vocab.TOPOLOGIES:
            import difflib

            close = difflib.get_close_matches(
                topology, sorted(vocab.TOPOLOGIES), n=1, cutoff=0.6
            )
            hint = f" — did you mean {close[0]!r}?" if close else ""
            self.error("body.topology", f"unrecognized topology {topology!r}{hint}: "
                                        "refusing to assemble a different surface "
                                        "than the document asks for")
        self.expect_float_array(body, "identity", vocab.IDENTITY_LENGTH, "body")
        self.expect_float_array(
            body, "resting_expression", vocab.RESTING_EXPRESSION_LENGTH, "body"
        )
        if "proportions" in body:
            self.check_proportions(body["proportions"])

    def check_proportions(self, proportions: object) -> None:
        """§4.3: every issue here is a hard error in every mode — like
        `topology` and `inputs`, a proportion misread changes what the
        document *builds*: a proportioned character silently assembled on
        the template skeleton is a different character than the file
        describes."""
        if not isinstance(proportions, dict):
            self.error("body.proportions", "must be an object")
            return
        for key, value in proportions.items():
            path = f"body.proportions.{key}"
            if key not in vocab.PROPORTION_NAMES:
                import difflib

                close = difflib.get_close_matches(
                    key, vocab.PROPORTION_NAMES, n=1, cutoff=0.6
                )
                hint = f" — did you mean {close[0]!r}?" if close else ""
                self.error(
                    path,
                    f"unknown proportion parameter{hint} (a proportion "
                    f"ignored is a different skeleton than the document "
                    f"describes)",
                )
                continue
            if not self.expect_number(value, path):
                continue
            # Compared at float32 — the format's canonical parameter
            # precision — so a canonicalized boundary value (float32(0.40)
            # is a hair above the decimal 0.40) round-trips as valid.
            import struct

            def _f32(v: float) -> float:
                return struct.unpack("f", struct.pack("f", v))[0]

            if abs(_f32(float(value))) > _f32(vocab.PROPORTION_LIMIT):
                self.error(
                    path,
                    f"out of range: |{float(value)}| exceeds the valid range "
                    f"±{vocab.PROPORTION_LIMIT} (out-of-range values are "
                    f"errors, never clamped)",
                )

    def check_recipe(self, recipe: object, path: str) -> None:
        if not isinstance(recipe, dict):
            self.error(path, "must be an object")
            return
        for reserved in vocab.RESERVED_RECIPE_FIELDS:
            if reserved in recipe:
                # Reserved for a future minor version, and — like an unknown
                # rig or topology — it changes what gets built: a recipe with
                # conditioning inputs cannot be honored by ignoring them.
                self.error(
                    f"{path}.{reserved}",
                    f"{reserved!r} is reserved for a future schema version; this "
                    f"implementation cannot honor conditioning inputs (SPEC.md §5.3)",
                )
        self.check_keys(
            recipe,
            {"component", "component_version", "prompt", "seed", "overrides"}
            | set(vocab.RESERVED_RECIPE_FIELDS),
            path,
        )
        self.expect_str(recipe, "component", path)
        self.expect_str(recipe, "component_version", path)
        self.expect_str(recipe, "prompt", path)
        self.expect_seed(recipe, path)
        overrides = recipe.get("overrides")
        if overrides is not None:
            if not isinstance(overrides, dict):
                self.error(f"{path}.overrides", "must be an object")
                return
            self.check_keys(overrides, {"steps", "guidance", "resolution"}, f"{path}.overrides")
            for key in ("steps", "resolution"):
                if key in overrides:
                    value = overrides[key]
                    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                        self.error(f"{path}.overrides.{key}", "must be a positive integer")
            if "guidance" in overrides:
                if self.expect_number(overrides["guidance"], f"{path}.overrides.guidance"):
                    if overrides["guidance"] <= 0:
                        self.error(f"{path}.overrides.guidance", "must be positive")

    def check_slot_key(self, slot: str, path: str) -> bool:
        """True if the key is a valid slot; emits the right issue otherwise.

        A wrong pluralization is the most likely authoring mistake and is a
        hard error in every mode, with the correction spelled out.
        """
        if slot in vocab.ALL_SLOTS:
            return True
        if slot in vocab.SLOT_MISTAKES:
            self.error(
                path,
                f"texture slot keys are singular — did you mean "
                f"{vocab.SLOT_MISTAKES[slot]!r}?",
            )
        else:
            self.unknown(path, "unknown texture slot")
        return False

    def check_slot_value(self, value: object, path: str) -> None:
        """A slot holds named maps. v0.1 defines one map, `albedo`; a flat
        recipe (recognized by its `component` key) is shorthand for it."""
        if value is None:
            self.error(path, "an unused optional slot must be omitted, not null")
            return
        if not isinstance(value, dict):
            self.error(path, "must be an object")
            return
        if "component" in value:            # flat shorthand: the albedo recipe
            self.check_recipe(value, path)
            return
        if "albedo" not in value:
            self.error(
                f"{path}.albedo",
                "every texture slot must define its albedo map (either a flat "
                "recipe or an explicit \"albedo\" entry)",
            )
        for map_name, recipe in value.items():
            map_path = f"{path}.{map_name}"
            if map_name not in vocab.MAPS:
                self.unknown(map_path, "unknown texture map")
                continue
            self.check_recipe(recipe, map_path)

    def check_textures(self, textures: object) -> None:
        if not isinstance(textures, dict):
            self.error("textures", "must be an object")
            return
        for slot in vocab.REQUIRED_SLOTS:
            if slot not in textures:
                self.error(f"textures.{slot}", "required slot is missing")
        for slot, value in textures.items():
            path = f"textures.{slot}"
            if self.check_slot_key(slot, path):
                self.check_slot_value(value, path)

    def check_hair(self, hair: object) -> None:
        if hair is None:
            return
        if not isinstance(hair, dict):
            self.error("hair", "must be an object or null")
            return
        allowed = {"schema_version", "seed", "family"} | set(vocab.HAIR_GROUPS)
        for key in hair:
            if key not in allowed:
                # SPEC.md §6: the hair block's vocabulary is closed; strict
                # validators reject unknown fields, default warns.
                self.unknown(f"hair.{key}", "unknown field in hair block")
        if hair.get("schema_version") != vocab.HAIR_SCHEMA_VERSION:
            self.error("hair.schema_version",
                       f"must be the integer {vocab.HAIR_SCHEMA_VERSION}")
        self.expect_seed(hair, "hair")
        self.expect_enum(hair, "family", vocab.HAIR_FAMILIES, "hair")

        for group, fields in vocab.HAIR_GROUPS.items():
            path = f"hair.{group}"
            block = hair.get(group)
            if block is None:
                self.error(path, "required field is missing")
                continue
            if not isinstance(block, dict):
                self.error(path, "must be an object")
                continue
            extra_allowed = {"rgb"} if group == "color" else set()
            for key in block:
                if key not in fields and key not in extra_allowed:
                    self.unknown(f"{path}.{key}", "unknown field in hair block")
            for field_name, values in fields.items():
                optional = (group, field_name) in vocab.HAIR_OPTIONAL_FIELDS
                if field_name not in block:
                    if not optional:
                        self.error(f"{path}.{field_name}", "required field is missing")
                    continue
                self.expect_enum(block, field_name, values, path)
            if group == "color":
                family = block.get("family")
                rgb = block.get("rgb")
                if family == "custom":
                    if rgb is None:
                        self.error(f"{path}.rgb", 'required when color.family is "custom"')
                elif rgb is not None:
                    self.error(f"{path}.rgb", 'only allowed when color.family is "custom"')
                if rgb is not None:
                    if not isinstance(rgb, list) or len(rgb) != 3:
                        self.error(f"{path}.rgb", "must be an array of 3 numbers")
                    else:
                        for i, item in enumerate(rgb):
                            if self.expect_number(item, f"{path}.rgb[{i}]"):
                                if not 0.0 <= item <= 1.0:
                                    self.error(f"{path}.rgb[{i}]", "must be in [0, 1]")

    def check_provenance(self, prov: object) -> None:
        if not isinstance(prov, dict):
            self.error("provenance", "must be an object")
            return
        self.check_keys(
            prov,
            {"prompt", "figure_prompt", "seed", "generator", "components",
             "created", "notes"},
            "provenance"
        )
        if "prompt" not in prov:
            self.error("provenance.prompt", "required field is missing (may be null)")
        elif prov["prompt"] is not None and not isinstance(prov["prompt"], str):
            self.error("provenance.prompt", "must be a string or null")
        if "figure_prompt" in prov and not isinstance(
                prov.get("figure_prompt"), str):
            self.error("provenance.figure_prompt", "must be a string")
        if "seed" in prov and (not isinstance(prov.get("seed"), int)
                               or isinstance(prov.get("seed"), bool)):
            self.error("provenance.seed", "must be an integer")
        self.expect_str(prov, "generator", "provenance")
        components = prov.get("components")
        if components is None:
            self.error("provenance.components", "required field is missing")
        elif not isinstance(components, dict):
            self.error("provenance.components", "must be an object")
        else:
            for name, ref in components.items():
                path = f"provenance.components.{name}"
                if not isinstance(ref, dict):
                    self.error(path, "must be an object")
                    continue
                self.check_keys(ref, {"version", "sha256"}, path)
                self.expect_str(ref, "version", path)
                sha = ref.get("sha256")
                if sha is not None and (not isinstance(sha, str) or not _SHA256_RE.match(sha)):
                    self.error(f"{path}.sha256", "must be 64 lowercase hex characters")
        created = prov.get("created")
        if created is not None and (
            not isinstance(created, str) or not _RFC3339_RE.match(created)
        ):
            self.error("provenance.created", "must be an RFC 3339 timestamp")
        notes = prov.get("notes")
        if notes is not None and not isinstance(notes, str):
            self.error("provenance.notes", "must be a string")

    def check_asset_descriptor(self, entry: object, path: str) -> None:
        if not isinstance(entry, dict):
            self.error(path, "must be an object")
            return
        self.check_keys(entry, {"sha256", "media_type", "width", "height"}, path)
        sha = entry.get("sha256")
        if not isinstance(sha, str) or not _SHA256_RE.match(sha):
            self.error(f"{path}.sha256", "must be 64 lowercase hex characters")
        self.expect_str(entry, "media_type", path)
        for key in ("width", "height"):
            value = entry.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                self.error(f"{path}.{key}", "must be a positive integer")

    def check_assets(self, assets: object) -> None:
        if not isinstance(assets, dict):
            self.error("assets", "must be an object")
            return
        for slot, entry in assets.items():
            path = f"assets.{slot}"
            if not self.check_slot_key(slot, path):
                continue
            if not isinstance(entry, dict):
                self.error(path, "must be an object")
                continue
            if "sha256" in entry:          # flat shorthand: the albedo descriptor
                self.check_asset_descriptor(entry, path)
                continue
            for map_name, descriptor in entry.items():
                map_path = f"{path}.{map_name}"
                if map_name not in vocab.MAPS:
                    self.unknown(map_path, "unknown texture map")
                    continue
                self.check_asset_descriptor(descriptor, map_path)


def hair_block_errors(hair: object) -> list[str]:
    """Validation errors for a hair block alone, as printable strings.

    The interpreter's repair loop checks a decoded hair block here before
    it enters a character document (ARCHITECTURE §2.2) — the decoding
    grammar is deliberately looser than the validator, so what the grammar
    admits must still pass this gate or be repaired.
    """
    checker = _Checker(strict=False)
    checker.check_hair(hair)
    return [f"{issue.path}: {issue.message}" for issue in checker.report.errors]


def validate_document(document: object, strict: bool = False) -> ValidationReport:
    """Validate a parsed character document. Returns a report; never raises."""
    checker = _Checker(strict)
    report = checker.report

    if not isinstance(document, dict):
        checker.error("$", "a character document must be a JSON object")
        return report

    checker.check_keys(
        document,
        {"format", "schema_version", "name", "body", "textures", "hair", "provenance", "assets"},
        "$",
    )

    if document.get("format") != vocab.FORMAT:
        checker.error("format", f'must be "{vocab.FORMAT}"')

    version = document.get("schema_version")
    if not isinstance(version, str) or not (match := _VERSION_RE.match(version)):
        checker.error("schema_version", 'must be a "<major>.<minor>" string')
    else:
        major, minor = int(match.group(1)), int(match.group(2))
        if major != vocab.SCHEMA_MAJOR or minor < vocab.SCHEMA_MINOR:
            checker.error(
                "schema_version",
                f"this implementation supports {vocab.SCHEMA_VERSION}; got {version}",
            )
        elif minor > vocab.SCHEMA_MINOR:
            checker.unknown(
                "schema_version",
                f"document uses newer minor version {version}; "
                f"unrecognized optional fields will be ignored",
            )

    if "name" in document and not isinstance(document["name"], str):
        checker.error("name", "must be a string")

    for key, check in (
        ("body", checker.check_body),
        ("textures", checker.check_textures),
        ("provenance", checker.check_provenance),
    ):
        if key not in document:
            checker.error(key, "required block is missing")
        else:
            check(document[key])

    if "hair" not in document:
        checker.error("hair", "required block is missing (may be null)")
    else:
        checker.check_hair(document["hair"])

    if "assets" in document:
        checker.check_assets(document["assets"])

    return report

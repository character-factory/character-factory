"""Generation-time geometry density controls.

Hair is generated *at* a target density rather than decimated after: a
decimator does not know which silhouette carries the style, and the
same reduction applied to a bob and to box braids produces very
different damage. These controls change what the generator emits.

Every field defaults to the engine's full-density behaviour, so a caller
that passes nothing gets exactly what it got before. Presets that select
non-default values are component data (see the provider), never
constants in code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["Density", "FULL"]


@dataclass(frozen=True)
class Density:
    """How finely the generator tessellates.

    `cap_u`/`cap_v` are the scalp cap's grid resolution. `count_scale`
    multiplies the number of visible locks in a clump field (never below
    two, the field's own floor). `profile_segments` and `guide_segments`
    are the cross-section and along-strand sample counts of a lock;
    `wisp_scale` multiplies the count of loose hairline strands.

    The braided controls are separate because braids are built from
    tubes rather than swept locks: `braid_radial_segments` overrides the
    tube cross-section, and `braid_spine_scale` multiplies the number of
    samples along each braid.
    """

    cap_u: int = 152
    cap_v: int = 56
    count_scale: float = 1.0
    profile_segments: int | None = None      # None keeps each spec's own
    guide_segments: int | None = None
    wisp_scale: float = 1.0
    braid_radial_segments: int | None = None
    braid_spine_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.cap_u < 8 or self.cap_v < 3:
            raise ValueError(
                f"cap resolution {self.cap_u}x{self.cap_v} is below the "
                f"generator's floor (8x3)")
        for name in ("count_scale", "wisp_scale", "braid_spine_scale"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        for name in ("profile_segments", "guide_segments",
                     "braid_radial_segments"):
            value = getattr(self, name)
            if value is not None and value < 3:
                raise ValueError(f"{name} must be at least 3, got {value}")

    @classmethod
    def from_mapping(cls, values: dict | None) -> "Density":
        """Build from component/config data, ignoring nothing silently:
        an unknown key is an error, because a preset whose typo is
        quietly dropped would ship at the wrong density."""
        if not values:
            return FULL
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                f"unknown hair density control(s): {', '.join(unknown)} "
                f"(known: {', '.join(sorted(known))})")
        return cls(**values)

    def scaled_count(self, count: int) -> int:
        """A clump field's lock count at this density (never below the
        field's own two-lock floor)."""
        return max(2, round(count * self.count_scale))

    def scaled_wisps(self, wisps: int) -> int:
        return max(0, round(wisps * self.wisp_scale))

    def apply_to_clump(self, spec):
        """A clump-field spec at this density."""
        changes = {"count": self.scaled_count(spec.count)}
        if self.profile_segments is not None:
            changes["profile_segments"] = self.profile_segments
        if self.guide_segments is not None:
            # The field validates its own floor; respect it rather than
            # raising, so a global preset can be shared across families.
            changes["guide_segments"] = max(8, self.guide_segments)
        return replace(spec, **changes)

    @property
    def is_full(self) -> bool:
        return self == FULL


FULL = Density()

"""Style = scalp under-cap + visible clumps + optional accents, composed."""

from dataclasses import dataclass, field

import numpy as np
import trimesh

from .braids import (
    BraidFieldSpec,
    BraidTailSpec,
    CornrowSpec,
    generate_braid_field,
    generate_braid_tail,
    generate_cornrows,
)
from .clumps import ClumpFieldSpec, generate_clump_field
from .head import Head
from .primitives import BangsSpec, BunSpec, MohawkSpec, TailSpec, generate_bangs, generate_bun, generate_mohawk, generate_tail
from .shell import CapSpec, HairlineSpec, generate_cap, generate_wisps


@dataclass
class Style:
    name: str = "custom"
    cap: CapSpec | None = None
    clump_field: ClumpFieldSpec | None = None
    clump_fields: tuple[ClumpFieldSpec, ...] = field(default_factory=tuple)
    bangs: BangsSpec | None = None
    tail: TailSpec | None = None
    bun: BunSpec | None = None
    mohawk: MohawkSpec | None = None
    braid_field: BraidFieldSpec | None = None
    cornrows: CornrowSpec | None = None
    braid_tail: BraidTailSpec | None = None


def generate(head: Head, style: Style) -> trimesh.Trimesh:
    parts = []
    if style.cap is not None:
        parts.append(generate_cap(head, style.cap))
        w = generate_wisps(head, style.cap)
        if w is not None:
            parts.append(w)
    if style.clump_field is not None:
        hl = style.cap.hairline if style.cap else HairlineSpec()
        parts.append(generate_clump_field(head, style.clump_field, hl))
    if style.clump_fields:
        hl = style.cap.hairline if style.cap else HairlineSpec()
        parts.extend(generate_clump_field(head, spec, hl) for spec in style.clump_fields)
    if style.bangs is not None:
        hl = style.cap.hairline if style.cap else None
        v_of_u = hl.v_of_u if hl else (lambda u: np.full_like(np.asarray(u, dtype=float), 0.68))
        vol = style.cap.volume if style.cap else 0.1
        parts.append(generate_bangs(head, style.bangs, v_of_u, vol))
    if style.tail is not None:
        parts.append(generate_tail(head, style.tail))
    if style.bun is not None:
        parts.append(generate_bun(head, style.bun))
    if style.mohawk is not None:
        parts.append(generate_mohawk(head, style.mohawk))
    if style.braid_field is not None or style.cornrows is not None:
        hl = style.cap.hairline if style.cap else HairlineSpec()
        if style.braid_field is not None:
            parts.append(generate_braid_field(head, style.braid_field, hl.v_of_u))
        if style.cornrows is not None:
            parts.append(generate_cornrows(head, style.cornrows, hl.v_of_u))
    if style.braid_tail is not None:
        parts.append(generate_braid_tail(head, style.braid_tail))
    if not parts:
        raise ValueError("empty style")
    if len(parts) == 1:
        return parts[0]
    uv = np.vstack([p.vertex_attributes["chart_uv"] for p in parts])
    out = trimesh.util.concatenate(parts)
    out.vertex_attributes["chart_uv"] = uv
    return out

"""Named style presets spanning the taxonomy + a seeded random sampler."""

import numpy as np

from .braids import BraidFieldSpec, BraidTailSpec, CornrowSpec
from .clumps import ClumpFieldSpec
from .guides import GuideDrapeSpec
from .primitives import BangsSpec, BunSpec, MohawkSpec, TailSpec
from .shell import CapSpec, HairlineSpec
from .style import Style
from .texture import TextureSpec


def _hl(**kw):
    return HairlineSpec(**kw)


PRESETS: dict[str, tuple[Style, TextureSpec]] = {
    "buzz": (
        Style(cap=CapSpec(volume=0.035, edge_frac=0.45, taper=0.0)),
        TextureSpec(melanin=0.9, curl=0.7, streak_contrast=0.3),
    ),
    "crop": (
        Style(cap=CapSpec(volume=0.09, edge_frac=0.25, lump_amp=0.012, lump_freq=14)),
        TextureSpec(melanin=0.8, curl=0.2),
    ),
    "curls_short": (
        Style(cap=CapSpec(volume=0.22, edge_frac=0.28, lift=0.35,
                          lump_amp=0.07, lump_freq=12)),
        TextureSpec(melanin=0.9, curl=1.0, streak_contrast=0.7),
    ),
    "afro": (
        Style(cap=CapSpec(volume=0.44, volume_back=0.48,
                          edge_frac=0.48, lift=0.14, taper=-0.06,
                          lump_amp=0.025, lump_freq=6, clump=0.0,
                          wisps=8, wisp_len=0.08,
                          hairline=_hl(front=0.78, mid=0.86, recess=1.02, knee=1.24, sideburn=1.40,
                                       post_ear=1.30, nape=1.38)),
              clump_field=ClumpFieldSpec(
                  mode="coily", count=68, volume=0.46, lift=0.052,
                  width=0.085, thickness=0.070, length=0.23,
                  wave=0.70, profile_segments=8, seed=47)),
        TextureSpec(melanin=0.91, redness=0.26, curl=0.65,
                    streak_contrast=0.18, normal_strength=0.20,
                    base_texture="assets/textures/generated/coily_dark_source.png",
                    base_texture_color=0.10, base_texture_detail=0.42),
    ),
    "pixie": (
        Style(cap=CapSpec(volume=0.13, edge_frac=0.2, lump_amp=0.02, lump_freq=7,
                          hairline=_hl(front=0.70, mid=0.79, recess=0.95, knee=1.24, sideburn=1.42,
                                       post_ear=1.32, nape=1.48)),
              bangs=BangsSpec(length=0.45, lift=0.10, tips="wispy")),
        TextureSpec(melanin=0.55, redness=0.4, wave=0.25),
    ),
    "bob": (
        Style(cap=CapSpec(volume=0.050, taper=0.05, lift=0.06,
                          clump=0.0, wisps=6, wisp_len=0.12,
                          part_u=0.03, part_depth=0.82, part_width=0.035),
              clump_field=ClumpFieldSpec(
                  mode="long", count=20, part_u=0.03, root_spread=0.76,
                  volume=0.09, lift=0.07, width=0.31, thickness=0.040,
                  lower_width=0.70,
                  length=1.04, length_front=0.84,
                  face_gap=0.30, flare=0.24, tuck=0.42, wave=0.18,
                  tip_jitter=0.06, seed=16)),
        TextureSpec(melanin=0.75, redness=0.3, streak_contrast=0.30,
                    normal_strength=0.32),
    ),
    "bob_bangs": (
        Style(cap=CapSpec(volume=0.17, taper=-0.3, lift=0.2,
                          curtain_len=0.9, curtain_len_front=0.8,
                          curtain_gap=0.32, curtain_in=0.35, curtain_flare=0.45,
                          lump_amp=0.015, lump_freq=6),
              bangs=BangsSpec(length=0.55, tips="wispy")),
        TextureSpec(melanin=0.85, redness=0.2),
    ),
    "long_straight": (
        Style(cap=CapSpec(volume=0.14, lift=0.15, curtain_len=3.8, curtain_len_front=2.6,
                          curtain_gap=0.30, curtain_flare=0.3, curtain_in=0.12)),
        TextureSpec(melanin=0.7, redness=0.3),
    ),
    "long_wavy": (
        Style(cap=CapSpec(volume=0.045, lift=0.06, clump=0.0,
                          wisps=8, wisp_len=0.24,
                          part_u=0.06, part_depth=0.94, part_width=0.046,
                          part_open=True, part_open_width=0.013,
                          hairline=_hl(front=0.72, mid=0.82, recess=0.98,
                                       knee=1.23, sideburn=1.40,
                                       post_ear=1.30, nape=1.40)),
              clump_field=ClumpFieldSpec(
                  mode="long", count=20, part_u=0.06, part_gap=0.064,
                  root_spread=0.82,
                  volume=0.115, lift=0.10,
                  width=0.30, thickness=0.032, lower_width=0.78, layers=2,
                  length=3.55, length_front=2.42, length_side=2.82, face_gap=0.32,
                  flare=0.27, tuck=0.40, wave=1.18, wave_freq=1.32,
                  tip_jitter=0.14, guide_segments=19, seed=31,
                  drape=GuideDrapeSpec(enabled=True, routing="split", stiffness=0.48)),
              clump_fields=(ClumpFieldSpec(
                  mode="long", count=12, part_u=0.06, part_gap=0.072,
                  root_spread=0.86, volume=0.075, lift=0.075,
                  width=0.51, thickness=0.034, lower_width=0.88,
                  length=3.42, length_front=2.32, length_side=2.76, face_gap=0.30,
                  flare=0.24, tuck=0.42, wave=0.90, wave_freq=1.18,
                  tip_jitter=0.07, guide_segments=19, seed=313,
                  drape=GuideDrapeSpec(enabled=True, routing="split", stiffness=0.52)),)),
        TextureSpec(melanin=0.68, redness=0.70, wave=0.45,
                    streak_contrast=0.26, lock_contrast=0.30, normal_strength=0.52,
                    base_texture="assets/textures/generated/auburn_flow_source.png",
                    base_texture_color=0.05, base_texture_detail=0.40),
    ),
    "side_part": (
        Style(cap=CapSpec(volume=0.035, edge_frac=0.30, taper=0.08,
                          clump=0.0, wisps=5, wisp_len=0.11,
                          part_u=0.28, part_depth=0.99, part_width=0.045,
                          part_open=True, part_open_width=0.012,
                          hairline=_hl(front=0.72, mid=0.81, recess=0.96,
                                       knee=1.22, sideburn=1.38,
                                       post_ear=1.28, nape=1.40)),
              clump_field=ClumpFieldSpec(
                  mode="short", count=14, part_u=0.28, part_gap=0.070,
                  root_spread=0.58, volume=0.050, lift=0.040,
                  width=0.31, thickness=0.035,
                  layers=2, layer_offset=0.010, hairline_margin=0.045,
                  seed=23)),
        TextureSpec(melanin=0.82, redness=0.25, streak_contrast=0.30,
                    normal_strength=0.30),
    ),
    "undercut": (
        Style(cap=CapSpec(volume=0.16, edge_frac=0.12, taper=0.1, part_u=-0.30,
                          part_depth=0.6,
                          hairline=_hl(front=0.70, mid=0.78, recess=0.86, knee=0.92, sideburn=0.95,
                                       post_ear=0.95, nape=1.00))),
        TextureSpec(melanin=0.78, redness=0.2, wave=0.3),
    ),
    "mohawk": (
        Style(cap=CapSpec(volume=0.03, edge_frac=0.5, taper=0.0),
              mohawk=MohawkSpec(height=0.6, thickness=0.3, spike=0.2)),
        TextureSpec(melanin=0.95, streak_contrast=0.4),
    ),
    "ponytail": (
        Style(cap=CapSpec(volume=0.10, edge_frac=0.12),
              tail=TailSpec(pos_v=0.5, length=2.2, sag=0.6)),
        TextureSpec(melanin=0.65, redness=0.35),
    ),
    "top_knot": (
        Style(cap=CapSpec(volume=0.09, edge_frac=0.12),
              bun=BunSpec(pos_v=0.16, radius=0.4, tube=0.17)),
        TextureSpec(melanin=0.9, redness=0.2),
    ),
    "low_bun": (
        Style(cap=CapSpec(volume=0.10, edge_frac=0.12),
              bun=BunSpec(pos_v=0.55, radius=0.45, tube=0.16)),
        TextureSpec(melanin=0.55, redness=0.3, grey=0.2),
    ),
    "box_braids": (
        Style(cap=CapSpec(volume=0.02, edge_frac=0.5, taper=0.0),
              braid_field=BraidFieldSpec(mode="braid", count=48, length=1.7)),
        TextureSpec(melanin=0.93, redness=0.2, curl=0.4, streak_contrast=0.4),
    ),
    "locs": (
        Style(cap=CapSpec(volume=0.02, edge_frac=0.5, taper=0.0),
              braid_field=BraidFieldSpec(mode="loc", count=60, length=1.3,
                                         strand_radius=0.055, knot_freq=5)),
        TextureSpec(melanin=0.9, redness=0.25, curl=0.5, streak_contrast=0.45),
    ),
    "twists_short": (
        Style(cap=CapSpec(volume=0.02, edge_frac=0.5, taper=0.0),
              braid_field=BraidFieldSpec(mode="twist", count=56, length=0.55,
                                         spread=0.5, strand_radius=0.05)),
        TextureSpec(melanin=0.92, redness=0.2, curl=0.7, streak_contrast=0.5),
    ),
    "cornrows": (
        Style(cornrows=CornrowSpec(rows=9)),
        TextureSpec(melanin=0.94, redness=0.2, streak_contrast=0.35),
    ),
    "braid_ponytail": (
        Style(cap=CapSpec(volume=0.10, edge_frac=0.12),
              braid_tail=BraidTailSpec(length=2.4)),
        TextureSpec(melanin=0.8, redness=0.3),
    ),
    # ---- fades / tapers / groomed short styles ----
    "fade": (
        Style(cap=CapSpec(volume=0.19, edge_frac=0.3, taper=0.0, clump=0.3,
                          fade_start_v=0.52, fade_end_v=0.92, fade_floor=0.04,
                          lump_amp=0.012, lump_freq=14)),
        TextureSpec(melanin=0.9, curl=0.6, streak_contrast=0.35),
    ),
    "taper_waves": (
        Style(cap=CapSpec(volume=0.055, edge_frac=0.35, taper=0.0,
                          fade_start_v=0.75, fade_end_v=1.1, fade_floor=0.25,
                          lump_amp=0.008, lump_freq=18)),
        TextureSpec(melanin=0.93, curl=0.9, streak_contrast=0.55),
    ),
    "slick_back": (
        Style(cap=CapSpec(volume=0.15, volume_back=0.21, edge_frac=0.2, taper=0.45, clump=0.25,
                          fade_start_v=0.68, fade_end_v=1.05, fade_floor=0.2)),
        TextureSpec(melanin=0.78, redness=0.25, streak_contrast=0.4),
    ),
    "comb_over": (
        Style(cap=CapSpec(volume=0.17, edge_frac=0.22, part_u=-0.32, part_depth=0.9,
                          part_width=0.035, side_bias=0.5, clump=0.3,
                          fade_start_v=0.72, fade_end_v=1.05, fade_floor=0.15)),
        TextureSpec(melanin=0.7, redness=0.3),
    ),
    "pompadour": (
        Style(cap=CapSpec(volume=0.14, edge_frac=0.2, clump=0.2,
                          fade_start_v=0.58, fade_end_v=0.96, fade_floor=0.06),
              bangs=BangsSpec(length=0.8, u_range=0.34, updo=1.0, tips="blunt")),
        TextureSpec(melanin=0.85, redness=0.2, wave=0.2),
    ),
    # ---- balding ----
    "receding": (
        Style(cap=CapSpec(volume=0.07, edge_frac=0.25,
                          hairline=_hl(front=0.58, mid=0.50, recess=0.42, knee=1.20, sideburn=1.42,
                                       post_ear=1.30, nape=1.44))),
        TextureSpec(melanin=0.55, redness=0.25, grey=0.3),
    ),
    "bald_crown": (
        Style(cap=CapSpec(volume=0.06, edge_frac=0.22, crown_gap_v=0.55,
                          hairline=_hl(front=0.74, mid=0.82, recess=0.98, knee=1.26, sideburn=1.44,
                                       post_ear=1.32, nape=1.48))),
        TextureSpec(melanin=0.45, redness=0.25, grey=0.55),
    ),
    # ---- compositions ----
    "pigtails": (
        Style(cap=CapSpec(volume=0.10, edge_frac=0.12, part_u=0.0, part_depth=0.45),
              tail=TailSpec(twin=True, pos_u=0.62, pos_v=0.72, length=1.6, sag=0.75)),
        TextureSpec(melanin=0.6, redness=0.4),
    ),
    "space_buns": (
        Style(cap=CapSpec(volume=0.09, edge_frac=0.12, part_u=0.0, part_depth=0.4),
              bun=BunSpec(twin=True, pos_u=0.5, pos_v=0.28, radius=0.34, tube=0.14)),
        TextureSpec(melanin=0.75, redness=0.3),
    ),
    "bantu_knots": (
        Style(cap=CapSpec(volume=0.03, edge_frac=0.45, taper=0.0),
              bun=BunSpec(count=9, radius=0.20, tube=0.09, turns=2.6)),
        TextureSpec(melanin=0.93, curl=0.8, streak_contrast=0.45),
    ),
    "man_bun": (
        Style(cap=CapSpec(volume=0.08, edge_frac=0.15,
                          fade_start_v=0.8, fade_end_v=1.15, fade_floor=0.3),
              bun=BunSpec(pos_v=0.3, radius=0.3, tube=0.15, turns=1.8)),
        TextureSpec(melanin=0.8, redness=0.3),
    ),
    # ---- gap-fill vs the real-world style taxonomy (Wikipedia list of
    # hairstyles + the protective-style taxonomy): silhouettes the 32-preset
    # v1 set could not reach at all, not just recolours of existing ones.
    "mullet": (
        Style(cap=CapSpec(volume=0.13, edge_frac=0.2, clump=0.35,
                          fade_start_v=0.62, fade_end_v=1.0, fade_floor=0.22,
                          curtain_len=1.5, curtain_len_front=0.0,
                          curtain_gap=0.66, curtain_flare=0.2, curtain_in=0.2,
                          tip_jag=0.3)),
        TextureSpec(melanin=0.62, redness=0.35, wave=0.3),
    ),
    "bowl_cut": (
        Style(cap=CapSpec(volume=0.20, taper=-0.2, edge_frac=0.10, clump=0.2,
                          hairline=_hl(front=0.86, mid=0.94, recess=1.06,
                                       knee=1.16, sideburn=1.20, post_ear=1.20,
                                       nape=1.26, jitter=0.006)),
              bangs=BangsSpec(length=0.30, u_range=0.40, lift=0.05, tips="blunt")),
        TextureSpec(melanin=0.88, streak_contrast=0.4),
    ),
    "shag": (
        Style(cap=CapSpec(volume=0.19, lift=0.25, clump=0.6, clump_count=13,
                          curtain_len=1.5, curtain_len_front=1.0,
                          curtain_gap=0.26, curtain_flare=0.5, tip_jag=0.42,
                          lump_amp=0.04, lump_freq=7),
              bangs=BangsSpec(length=0.5, u_range=0.30, tips="wispy")),
        TextureSpec(melanin=0.6, redness=0.4, wave=0.6),
    ),
    "wolf_cut": (
        Style(cap=CapSpec(volume=0.22, lift=0.4, clump=0.65, clump_count=12,
                          curtain_len=2.1, curtain_len_front=1.2,
                          curtain_gap=0.24, curtain_flare=0.6, tip_jag=0.45,
                          lump_amp=0.05, lump_freq=6),
              bangs=BangsSpec(length=0.55, u_range=0.34, tips="wispy")),
        TextureSpec(melanin=0.8, wave=0.7, curl=0.2),
    ),
    "hime": (
        Style(cap=CapSpec(volume=0.15, clump=0.3, clump_count=20,
                          curtain_len=3.6, curtain_len_front=1.15,
                          curtain_gap=0.22, curtain_flare=0.22, curtain_in=0.1,
                          tip_jag=0.02),
              bangs=BangsSpec(length=0.52, u_range=0.36, tips="blunt")),
        TextureSpec(melanin=0.95, streak_contrast=0.5),
    ),
    "curtain_bangs": (
        Style(cap=CapSpec(volume=0.17, lift=0.2, part_u=0.0, part_depth=0.55,
                          part_width=0.05, clump=0.5,
                          curtain_len=2.4, curtain_len_front=1.7,
                          curtain_gap=0.26, curtain_flare=0.4, tip_jag=0.22),
              bangs=BangsSpec(length=0.6, u_range=0.30, lift=0.18, tips="wispy")),
        TextureSpec(melanin=0.55, redness=0.4, wave=0.5),
    ),
    "beehive": (
        Style(cap=CapSpec(volume=0.55, lift=1.3, taper=-0.5, edge_frac=0.30,
                          clump=0.22, clump_count=24,
                          hairline=_hl(front=0.62, mid=0.72, recess=0.92,
                                       knee=1.24, sideburn=1.38, post_ear=1.28,
                                       nape=1.36))),
        TextureSpec(melanin=0.85, streak_contrast=0.5),
    ),
    "flat_top": (
        Style(cap=CapSpec(volume=0.42, taper=-0.75, edge_frac=0.55, clump=0.12,
                          fade_start_v=0.72, fade_end_v=1.0, fade_floor=0.05)),
        TextureSpec(melanin=0.95, curl=1.0, streak_contrast=0.35),
    ),
    "afro_puffs": (
        Style(cap=CapSpec(volume=0.05, edge_frac=0.4, taper=0.0, clump=0.0,
                          part_u=0.0, part_depth=0.5),
              bun=BunSpec(twin=True, pos_u=0.55, pos_v=0.38, radius=0.62,
                          tube=0.30, turns=1.5)),
        TextureSpec(melanin=0.92, curl=1.0, streak_contrast=0.5),
    ),
    "liberty_spikes": (
        Style(cap=CapSpec(volume=0.03, edge_frac=0.5, taper=0.0, clump=0.0),
              mohawk=MohawkSpec(height=1.15, thickness=0.22, spike=0.85)),
        TextureSpec(melanin=0.3, dye=(0.85, 0.2, 0.25), dye_amount=0.9),
    ),
    "senegalese_twists": (
        Style(cap=CapSpec(volume=0.02, edge_frac=0.5, taper=0.0, clump=0.0),
              braid_field=BraidFieldSpec(mode="twist", count=44, length=2.6,
                                         strand_radius=0.045, knot_freq=7,
                                         spread=0.28)),
        TextureSpec(melanin=0.9, redness=0.25, streak_contrast=0.45),
    ),
    "knotless_braids": (
        Style(cap=CapSpec(volume=0.02, edge_frac=0.5, taper=0.0, clump=0.0),
              braid_field=BraidFieldSpec(mode="braid", count=64, length=2.9,
                                         width=0.038, strand_radius=0.033,
                                         knot_freq=11, spread=0.25)),
        TextureSpec(melanin=0.93, redness=0.2, streak_contrast=0.4),
    ),
    "faux_locs": (
        Style(cap=CapSpec(volume=0.02, edge_frac=0.5, taper=0.0, clump=0.0),
              braid_field=BraidFieldSpec(mode="loc", count=52, length=2.8,
                                         strand_radius=0.062, knot_freq=4)),
        TextureSpec(melanin=0.88, redness=0.3, curl=0.4, streak_contrast=0.45),
    ),
    "fulani_braids": (
        Style(cap=CapSpec(volume=0.02, edge_frac=0.5, taper=0.0, clump=0.0),
              cornrows=CornrowSpec(rows=7, tail_length=1.6),
              braid_field=BraidFieldSpec(mode="braid", count=22, length=2.2,
                                         width=0.05, strand_radius=0.042,
                                         beads=0.5)),
        TextureSpec(melanin=0.9, redness=0.25, streak_contrast=0.45),
    ),
    "half_up": (
        Style(cap=CapSpec(volume=0.12, curtain_len=2.6, curtain_len_front=1.9,
                          curtain_gap=0.28, curtain_in=0.18),
              bun=BunSpec(pos_v=0.22, radius=0.28, tube=0.13, turns=1.8)),
        TextureSpec(melanin=0.5, redness=0.5, wave=0.5, ombre=(0.85, 0.7, 0.5),
                    ombre_start=0.45),
    ),
}


def sample_style(seed: int) -> tuple[str, Style, TextureSpec]:
    """Seeded random style: pick an archetype, jitter its parameters."""
    rng = np.random.default_rng(seed)
    name = list(PRESETS)[rng.integers(len(PRESETS))]
    style, tex = PRESETS[name]
    import copy

    style = copy.deepcopy(style)
    tex = copy.deepcopy(tex)
    c = style.cap
    if c is not None:
        c.volume = float(np.clip(c.volume * rng.uniform(0.75, 1.35), 0.02, 0.75))
        c.lump_amp = float(max(0.0, c.lump_amp * rng.uniform(0.5, 1.8)))
        c.seed = int(rng.integers(1 << 30))
        if c.curtain_len > 0:
            c.curtain_len = float(c.curtain_len * rng.uniform(0.7, 1.3))
            c.curtain_len_front = float(c.curtain_len_front * rng.uniform(0.7, 1.2))
        if rng.random() < 0.35 and c.curtain_len == 0:
            c.part_u = float(rng.uniform(-0.35, 0.35))
    if style.tail is not None:
        style.tail.length *= float(rng.uniform(0.6, 1.4))
        style.tail.pos_v = float(np.clip(style.tail.pos_v + rng.uniform(-0.15, 0.25), 0.15, 0.8))
    if style.bun is not None:
        style.bun.radius *= float(rng.uniform(0.8, 1.25))
    if style.braid_field is not None:
        bf = style.braid_field
        bf.count = int(bf.count * rng.uniform(0.7, 1.4))
        bf.length = float(bf.length * rng.uniform(0.6, 1.5))
        bf.strand_radius = float(bf.strand_radius * rng.uniform(0.8, 1.3))
        bf.seed = int(rng.integers(1 << 30))
    if style.cornrows is not None:
        style.cornrows.rows = int(np.clip(style.cornrows.rows + rng.integers(-2, 4), 5, 14))
        style.cornrows.seed = int(rng.integers(1 << 30))
    if style.braid_tail is not None:
        style.braid_tail.length *= float(rng.uniform(0.7, 1.3))
    clump_specs = (() if style.clump_field is None else (style.clump_field,)) + tuple(style.clump_fields)
    for cf in clump_specs:
        cf.width = float(cf.width * rng.uniform(0.86, 1.16))
        cf.lift = float(cf.lift * rng.uniform(0.75, 1.30))
        cf.seed = int(rng.integers(1 << 30))
        if cf.mode == "long":
            cf.length = float(cf.length * rng.uniform(0.75, 1.28))
            cf.length_front = float(cf.length_front * rng.uniform(0.80, 1.20))
            cf.wave = float(np.clip(cf.wave + rng.uniform(-0.18, 0.25), 0, 1.2))
    if c is not None and rng.random() < 0.4:
        c.side_bias = float(rng.uniform(-0.45, 0.45))  # asymmetry
    tex.melanin = float(np.clip(rng.beta(2.2, 1.1), 0.05, 0.98))
    tex.redness = float(rng.beta(1.4, 3.0))
    tex.wave = float(np.clip(tex.wave + rng.uniform(-0.2, 0.4), 0, 1.2))
    tex.curl = float(np.clip(tex.curl + rng.uniform(-0.2, 0.3), 0, 1.2))
    tex.grey = float(max(0.0, rng.uniform(-0.4, 0.35)))
    r = rng.random()
    if r < 0.10:  # fashion dye
        tex.dye = tuple(rng.uniform(0.05, 0.9, 3).round(3))
        tex.dye_amount = float(rng.uniform(0.7, 1.0))
    elif r < 0.22:  # ombre tips
        tex.ombre = tuple(rng.uniform(0.2, 0.95, 3).round(3))
        tex.ombre_start = float(rng.uniform(0.3, 0.6))
    elif r < 0.32:  # highlights
        tex.highlight = tuple((rng.uniform(0.5, 0.95, 3)).round(3))
        tex.highlight_frac = float(rng.uniform(0.08, 0.3))
    tex.seed = int(rng.integers(1 << 30))
    return f"{name}_s{seed}", style, tex

"""Hair density controls: defaults preserve full density, presets are data."""

import pytest

from character_factory.hair import WigProvider
from character_factory.hair.wig.density import FULL, Density


def test_defaults_are_full_density():
    assert FULL.is_full
    assert Density().cap_u == 152 and Density().cap_v == 56
    assert Density().count_scale == 1.0 and Density().wisp_scale == 1.0
    # None means "whatever the spec itself says" — the generator's own values.
    assert Density().profile_segments is None
    assert Density().guide_segments is None


def test_controls_scale_counts_with_floors():
    budget = Density(count_scale=0.38, wisp_scale=0.25)
    assert budget.scaled_count(18) == 7
    assert budget.scaled_count(3) == 2      # never below the field's floor
    assert budget.scaled_wisps(40) == 10
    assert budget.scaled_wisps(1) == 0


def test_clump_spec_takes_the_density():
    from character_factory.hair.wig.clumps import ClumpFieldSpec

    spec = ClumpFieldSpec(count=20, profile_segments=8, guide_segments=19)
    applied = Density(count_scale=0.38, profile_segments=3,
                      guide_segments=8).apply_to_clump(spec)
    assert applied.count == 8
    assert applied.profile_segments == 3
    assert applied.guide_segments == 8
    assert spec.count == 20                # the source spec is untouched


def test_guide_segments_respect_the_generators_floor():
    from character_factory.hair.wig.clumps import ClumpFieldSpec

    applied = Density(guide_segments=4).apply_to_clump(ClumpFieldSpec())
    assert applied.guide_segments == 8


@pytest.mark.parametrize("controls", [
    {"cap_u": 4, "cap_v": 12},              # below the generator's floor
    {"count_scale": 1.5},                   # scales are fractions
    {"profile_segments": 2},                # a profile needs three sides
])
def test_impossible_densities_are_rejected(controls):
    with pytest.raises(ValueError):
        Density(**controls)


def test_unknown_preset_key_is_an_error_not_a_silent_drop():
    # A typo in component data must not ship geometry at the wrong density.
    with pytest.raises(ValueError, match="unknown hair density control"):
        Density.from_mapping({"cap_u": 35, "cap_vv": 12})


def test_provider_defaults_to_full_density():
    density, name = WigProvider().density_for("bob")
    assert density.is_full and name is None


def test_provider_selects_the_family_preset_from_component_data():
    presets = {"budget": {"cap_u": 35, "cap_v": 12, "count_scale": 0.38},
               "families": {"bob": "budget"}}
    provider = WigProvider(density_presets=presets)
    density, name = provider.density_for("bob")
    assert name == "budget" and density.cap_u == 35
    # A family absent from the map generates at full density — which is how
    # braided families stay on the declared higher tier.
    other, other_name = provider.density_for("braids")
    assert other.is_full and other_name is None


def test_preset_named_but_undeclared_is_an_error():
    provider = WigProvider(density_presets={"families": {"bob": "missing"}})
    with pytest.raises(ValueError, match="not declared by the component"):
        provider.density_for("bob")

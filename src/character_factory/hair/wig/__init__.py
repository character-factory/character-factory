"""make-wig — head-agnostic parametric hair generator (opaque textured geometry)."""

from .head import Head
from .clumps import ClumpFieldSpec, generate_clump_field
from .guides import GuideDrapeSpec
from .intent import (
    HairIntent,
    HairIntentError,
    HairlineIntent,
    HairPlan,
    compile_hair_intent,
    hair_intent_schema,
    load_hair_intent,
)
from .shell import generate_cap

__all__ = [
    "Head", "ClumpFieldSpec", "GuideDrapeSpec", "HairIntent", "HairIntentError", "HairlineIntent",
    "HairPlan", "compile_hair_intent", "hair_intent_schema", "load_hair_intent", "generate_cap",
    "generate_clump_field",
]

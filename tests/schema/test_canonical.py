"""Canonical form (RFC 8785) and content identity."""

import json
import struct

import pytest

from character_factory.schema.canonical import (
    canonical_form,
    content_id,
    float32_value,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, "1"),
        (-0.0, "0"),
        (0.5, "0.5"),
        (-2.75, "-2.75"),
        (1e21, "1e+21"),
        (1e16, "10000000000000000"),
        (1e-7, "1e-7"),
        (0.00001, "0.00001"),
        (0.002, "0.002"),
        (1e30, "1e+30"),
        (1e-27, "1e-27"),
        # Shortest round-trip form, per ECMAScript Number::toString: the
        # 16-digit form already parses back to the identical double.
        (333333333.33333329, "333333333.3333333"),
        (7, "7"),
        (-42, "-42"),
    ],
)
def test_number_serialization(value, expected):
    assert canonical_form([value]) == f"[{expected}]".encode()


def test_number_vector():
    doc = {"numbers": [333333333.33333329, 1e30, 4.5, 2e-3, 1e-27]}
    assert (
        canonical_form(doc)
        == b'{"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27]}'
    )


def test_nan_and_infinity_rejected():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            canonical_form({"x": bad})


def test_key_ordering_and_compactness():
    doc = {"b": 1, "a": {"z": True, "m": None}, "list": [1, "two", False]}
    assert canonical_form(doc) == b'{"a":{"m":null,"z":true},"b":1,"list":[1,"two",false]}'


def test_string_escapes():
    assert canonical_form(["a\"b\\c\n\t\x01"]) == b'["a\\"b\\\\c\\n\\t\\u0001"]'


def test_unicode_passthrough():
    # Non-control characters are emitted literally as UTF-8, not escaped.
    assert canonical_form(["héllo ✓"]) == '["héllo ✓"]'.encode("utf-8")


def test_content_id_invariant_to_formatting_and_key_order():
    a = json.loads('{"x": 1, "y": [1.0, 2.5]}')
    b = json.loads('{ "y":[1.0,2.5],   "x":1 }')
    assert content_id(a) == content_id(b)
    assert len(content_id(a)) == 64


def test_float32_value_round_trip():
    for x in (0.1, 1 / 3, 1234.5678, -0.000123, 3.14159265358979):
        f32 = float32_value(x)
        # Idempotent, and the emitted decimal parses back to the same float32.
        assert float32_value(f32) == f32
        emitted = canonical_form([f32])[1:-1].decode()
        reparsed = float(emitted)
        assert struct.pack("<f", reparsed) == struct.pack("<f", f32)

"""Canonical serialization and content identity for character documents.

Implements the JSON Canonicalization Scheme (RFC 8785): object keys sorted by
UTF-16 code units, no insignificant whitespace, minimal string escapes, and
numbers serialized with ECMAScript `Number::toString` semantics. The content
ID of a document is the SHA-256 of its canonical form (SPEC.md §2.1).

Stdlib only, by design.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

__all__ = ["canonical_form", "content_id", "float32_value"]


def float32_value(x: float) -> float:
    """The double equal to `x` rounded to IEEE 754 binary32.

    The format stores float32 values (SPEC.md §2); passing numbers through
    this before serialization guarantees the emitted decimal round-trips to
    the same float32.
    """
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def _es_number(x: float) -> str:
    """Serialize a finite number as ECMAScript `Number::toString(10)` would."""
    if isinstance(x, int):
        return str(x)
    if math.isnan(x) or math.isinf(x):
        raise ValueError("NaN and infinities are not valid in character documents")
    if x == 0:
        return "0"  # covers -0.0: ECMAScript String(-0) is "0"

    sign = "-" if x < 0 else ""
    # repr() gives the shortest digit string that round-trips the double.
    mantissa, _, exp_part = repr(abs(x)).partition("e")
    exp = int(exp_part) if exp_part else 0
    int_part, _, frac_part = mantissa.partition(".")
    digits = (int_part + frac_part).lstrip("0")
    # n: position of the decimal point relative to the digit string.
    n = len(int_part) - (len(int_part + frac_part) - len(digits)) + exp
    digits = digits.rstrip("0")
    k = len(digits)

    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    exponent = n - 1
    head = digits[0] + ("." + digits[1:] if k > 1 else "")
    return f"{sign}{head}e{'+' if exponent >= 0 else '-'}{abs(exponent)}"


_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _string(s: str) -> str:
    out = ['"']
    for ch in s:
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _utf16_key(s: str) -> bytes:
    return s.encode("utf-16-be")


def _serialize(value: Any, out: list[str]) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, str):
        out.append(_string(value))
    elif isinstance(value, (int, float)):
        out.append(_es_number(value))
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for i, item in enumerate(value):
            if i:
                out.append(",")
            _serialize(item, out)
        out.append("]")
    elif isinstance(value, dict):
        out.append("{")
        for i, key in enumerate(sorted(value, key=_utf16_key)):
            if not isinstance(key, str):
                raise TypeError(f"non-string object key: {key!r}")
            if i:
                out.append(",")
            out.append(_string(key))
            out.append(":")
            _serialize(value[key], out)
        out.append("}")
    else:
        raise TypeError(f"type not representable in JSON: {type(value).__name__}")


def canonical_form(document: dict) -> bytes:
    """RFC 8785 canonical serialization of a JSON-shaped document, as UTF-8."""
    out: list[str] = []
    _serialize(document, out)
    return "".join(out).encode("utf-8")


def content_id(document: dict) -> str:
    """Lowercase hex SHA-256 of the canonical form (SPEC.md §2.1)."""
    return hashlib.sha256(canonical_form(document)).hexdigest()

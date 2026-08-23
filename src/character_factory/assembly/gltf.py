"""A deterministic glTF 2.0 binary writer, and the reader used to verify it.

Written by hand on purpose: assembly promises byte-identical output for
identical inputs (SPEC.md §9), so the exporter controls every byte — stable
ordering, no timestamps, 4-byte-aligned buffer views, explicit min/max on
POSITION accessors, proper buffer targets. The reader exists so validation
re-parses the artifact from disk instead of trusting in-memory state.
"""

from __future__ import annotations

import json
import struct

import numpy as np

__all__ = ["GlbWriter", "component_count", "parse_glb", "read_accessor"]

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963

_COMPONENT_TYPES = {
    np.dtype(np.float32): 5126,
    np.dtype(np.uint32): 5125,
    np.dtype(np.uint16): 5123,
    np.dtype(np.uint8): 5121,
}
_TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_DTYPES_BY_CODE = {code: dtype for dtype, code in _COMPONENT_TYPES.items()}


def component_count(accessor_type: str) -> int:
    return _TYPE_SIZES[accessor_type]


class GlbWriter:
    """Accumulates binary views and accessors, then emits one .glb blob."""

    def __init__(self) -> None:
        self._blob = bytearray()
        self.buffer_views: list[dict] = []
        self.accessors: list[dict] = []

    def add_view(self, data: bytes, target: int | None = None) -> int:
        offset = len(self._blob)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self._blob.extend(data)
        self._blob.extend(b"\x00" * (-len(self._blob) % 4))
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def add_accessor(
        self,
        array: np.ndarray,
        accessor_type: str,
        *,
        target: int | None = None,
        minmax: bool = False,
    ) -> int:
        dtype = array.dtype
        if dtype not in _COMPONENT_TYPES:
            raise TypeError(f"unsupported accessor dtype {dtype}")
        width = _TYPE_SIZES[accessor_type]
        flat = array.reshape(-1, width) if width > 1 else array.reshape(-1, 1)
        accessor = {
            "bufferView": self.add_view(np.ascontiguousarray(array).tobytes(), target),
            "componentType": _COMPONENT_TYPES[dtype],
            "count": flat.shape[0],
            "type": accessor_type,
        }
        if minmax:
            accessor["min"] = [float(v) for v in flat.min(axis=0)]
            accessor["max"] = [float(v) for v in flat.max(axis=0)]
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def add_sparse_accessor(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        count: int,
        accessor_type: str,
        *,
        minmax: bool = False,
    ) -> int:
        """A sparse accessor over an implicit all-zeros base — the natural
        encoding for morph-target deltas, where most vertices do not move.
        `indices` must be sorted, unique uint32; `values` the matching rows."""
        if indices.dtype != np.dtype(np.uint32):
            raise TypeError("sparse indices must be uint32")
        if len(indices) != len(values):
            raise ValueError("sparse indices and values disagree in length")
        if len(indices) and not bool(np.all(np.diff(indices.astype(np.int64)) > 0)):
            raise ValueError("sparse indices must be strictly increasing")
        width = _TYPE_SIZES[accessor_type]
        accessor = {
            "componentType": _COMPONENT_TYPES[values.dtype],
            "count": count,
            "type": accessor_type,
            "sparse": {
                "count": int(len(indices)),
                "indices": {
                    "bufferView": self.add_view(
                        np.ascontiguousarray(indices).tobytes()
                    ),
                    "componentType": _COMPONENT_TYPES[np.dtype(np.uint32)],
                },
                "values": {
                    "bufferView": self.add_view(
                        np.ascontiguousarray(values).tobytes()
                    ),
                },
            },
        }
        if minmax:
            flat = values.reshape(-1, width) if width > 1 else values.reshape(-1, 1)
            if len(flat) == 0:
                flat = np.zeros((1, width), dtype=values.dtype)
            # The implicit base is zero, so min/max must include 0.
            accessor["min"] = [min(float(v), 0.0) for v in flat.min(axis=0)]
            accessor["max"] = [max(float(v), 0.0) for v in flat.max(axis=0)]
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def add_image(self, png_bytes: bytes) -> dict:
        """An embedded PNG: returns the glTF image object referencing a view."""
        return {"bufferView": self.add_view(png_bytes), "mimeType": "image/png"}

    def finish(self, gltf: dict) -> bytes:
        gltf = dict(gltf)
        gltf["buffers"] = [{"byteLength": len(self._blob)}]
        gltf["bufferViews"] = self.buffer_views
        gltf["accessors"] = self.accessors
        json_bytes = json.dumps(gltf, separators=(",", ":"), allow_nan=False).encode()
        json_bytes += b" " * (-len(json_bytes) % 4)
        bin_bytes = bytes(self._blob)
        total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
        return b"".join(
            [
                struct.pack("<III", GLB_MAGIC, 2, total),
                struct.pack("<II", len(json_bytes), CHUNK_JSON),
                json_bytes,
                struct.pack("<II", len(bin_bytes), CHUNK_BIN),
                bin_bytes,
            ]
        )


def parse_glb(data: bytes) -> tuple[dict, bytes]:
    magic, version, total = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2:
        raise ValueError("not a glTF 2.0 binary")
    if total != len(data):
        raise ValueError(f"glb header says {total} bytes, file has {len(data)}")
    offset = 12
    gltf: dict | None = None
    binary = b""
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8 : offset + 8 + length]
        if kind == CHUNK_JSON:
            gltf = json.loads(chunk.decode("utf-8"))
        elif kind == CHUNK_BIN:
            binary = chunk
        offset += 8 + length
    if gltf is None:
        raise ValueError("glb has no JSON chunk")
    return gltf, binary


def read_accessor(gltf: dict, binary: bytes, index: int) -> np.ndarray:
    accessor = gltf["accessors"][index]
    dtype = _DTYPES_BY_CODE[accessor["componentType"]]
    width = _TYPE_SIZES[accessor["type"]]

    def view_array(view_index: int, count: int, item_dtype) -> np.ndarray:
        view = gltf["bufferViews"][view_index]
        return np.frombuffer(
            binary, dtype=item_dtype, count=count,
            offset=view.get("byteOffset", 0),
        )

    if "bufferView" in accessor:
        view = gltf["bufferViews"][accessor["bufferView"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        dense = np.frombuffer(
            binary, dtype=dtype, count=accessor["count"] * width, offset=start
        ).reshape(accessor["count"], width)
    else:
        dense = np.zeros((accessor["count"], width), dtype=dtype)

    sparse = accessor.get("sparse")
    if sparse is not None:
        n = sparse["count"]
        idx_dtype = _DTYPES_BY_CODE[sparse["indices"]["componentType"]]
        indices = view_array(sparse["indices"]["bufferView"], n, idx_dtype)
        values = view_array(
            sparse["values"]["bufferView"], n * width, dtype
        ).reshape(n, width)
        dense = dense.copy()
        dense[indices.astype(np.int64)] = values
    return dense if width > 1 else dense.reshape(-1)

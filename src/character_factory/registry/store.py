"""Local component store: cache layout, fetching, and integrity.

Downloads use plain stdlib HTTP against Hugging Face's stable resolve URLs —
no client library, so the base install stays dependency-free. Every artifact
is SHA-256 verified after download and before first use; a mismatch is a
hard error, never a warning (ARCHITECTURE.md §4.2).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from character_factory.registry.model import ComponentEntry, RegistryError

__all__ = [
    "ComponentNotPublished", "IntegrityError", "cache_dir", "ensure_component",
    "missing_bytes",
]

_CHUNK = 1 << 20


class ComponentNotPublished(RegistryError):
    """The registry knows this component but its weights are not published yet."""


class IntegrityError(RegistryError):
    """A downloaded or cached artifact does not match its pinned SHA-256."""


def cache_dir() -> Path:
    """The local component cache root.

    Resolution order: ``CHARACTER_FACTORY_HOME``, then ``XDG_CACHE_HOME``,
    then ``~/.cache`` — always ending in ``character-factory/``.
    """
    override = os.environ.get("CHARACTER_FACTORY_HOME")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "character-factory"


def component_dir(entry: ComponentEntry) -> Path:
    return cache_dir() / "components" / entry.name / str(entry.version)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_url(entry: ComponentEntry, artifact_path: str) -> str:
    source = entry.source
    if source is None:
        raise ComponentNotPublished(
            f"component {entry.ref} is not published yet: its registry entry has "
            f"no source repository"
        )
    return (
        f"https://huggingface.co/{source['hf_repo']}/resolve/"
        f"{source['revision']}/{artifact_path}"
    )


def _request_headers(headers: dict[str, str] | None) -> dict[str, str]:
    return {"User-Agent": "character-factory", **(headers or {})}


def _download(
    url: str, target: Path, expected_bytes: int,
    headers: dict[str, str] | None = None,
    progress: Callable[[int], None] | None = None,
) -> None:
    """Fetch `url` to `target`; `progress` (if given) receives each chunk's
    byte count as it lands — and may raise to abandon the download."""
    request = urllib.request.Request(url, headers=_request_headers(headers))
    try:
        with urllib.request.urlopen(request) as response, target.open("wb") as out:
            if progress is None:
                shutil.copyfileobj(response, out, _CHUNK)
            else:
                while chunk := response.read(_CHUNK):
                    out.write(chunk)
                    progress(len(chunk))
    except urllib.error.URLError as error:
        raise RegistryError(f"download failed: {url} ({error})") from error
    actual = target.stat().st_size
    if actual != expected_bytes:
        raise IntegrityError(
            f"{url}: expected {expected_bytes} bytes, received {actual}"
        )


def fetch_json(url: str, headers: dict[str, str] | None = None) -> dict:
    """Fetch a JSON document (an alternate registry index) over HTTPS,
    applying the configured auth headers."""
    request = urllib.request.Request(url, headers=_request_headers(headers))
    try:
        with urllib.request.urlopen(request) as response:
            import json

            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RegistryError(f"registry index fetch failed: {url} ({error})") from error


def missing_bytes(entry: ComponentEntry) -> int:
    """Bytes of `entry`'s artifacts not yet in the cache (existence only —
    integrity is checked when the component is provisioned)."""
    target_dir = component_dir(entry)
    return sum(
        artifact["bytes"] for artifact in entry.artifacts
        if not (target_dir / artifact["path"]).is_file()
    )


def ensure_component(
    entry: ComponentEntry,
    *,
    fetch: Callable[[str, Path, int], None] | None = None,
    headers: dict[str, str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Return the local directory holding `entry`'s verified artifacts,
    downloading whatever is missing.

    `fetch` is injectable for tests; the default downloads over HTTPS with
    the configured auth headers applied (private staging repositories work
    authenticated, then identically unauthenticated after a public flip).
    `progress`, if given, is called with (bytes received so far, bytes to
    fetch in total) as the default fetch streams — a job can show a
    19 GB model arriving, and may raise from the callback to abandon it.
    """
    if fetch is None:
        total = missing_bytes(entry)
        received = 0

        def fetch(url: str, target: Path, expected: int) -> None:  # noqa: E731
            def advance(count: int) -> None:
                nonlocal received
                received += count
                progress(received, total)

            if progress is None:
                _download(url, target, expected, headers)
            else:
                _download(url, target, expected, headers, progress=advance)
    target_dir = component_dir(entry)
    # An empty artifact list must not "succeed" into a directory that does
    # not exist: unless the component is already provisioned locally, a
    # declared-but-unpublished entry (artifact lists are completed at
    # publish) fails here with the same clear error a source-less fetch
    # would give, instead of somewhere downstream.
    if not entry.artifacts and not target_dir.is_dir():
        raise ComponentNotPublished(
            f"component {entry.ref} is not published yet: its registry "
            f"entry declares no artifacts and it is not provisioned locally"
        )
    for artifact in entry.artifacts:
        path = target_dir / artifact["path"]
        if path.is_file():
            if _sha256_of(path) != artifact["sha256"]:
                raise IntegrityError(
                    f"{entry.ref}: cached artifact {artifact['path']} fails its "
                    f"integrity check; delete {path} and retry"
                )
            continue
        url = resolve_url(entry, artifact["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", suffix=".part", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            fetch(url, tmp_path, artifact["bytes"])
            actual = _sha256_of(tmp_path)
            if actual != artifact["sha256"]:
                raise IntegrityError(
                    f"{entry.ref}: {artifact['path']} downloaded with sha256 "
                    f"{actual}, registry pins {artifact['sha256']}"
                )
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)
    return target_dir

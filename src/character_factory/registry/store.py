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

__all__ = ["ComponentNotPublished", "IntegrityError", "cache_dir", "ensure_component"]

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


def _download(url: str, target: Path, expected_bytes: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "character-factory"})
    try:
        with urllib.request.urlopen(request) as response, target.open("wb") as out:
            shutil.copyfileobj(response, out, _CHUNK)
    except urllib.error.URLError as error:
        raise RegistryError(f"download failed: {url} ({error})") from error
    actual = target.stat().st_size
    if actual != expected_bytes:
        raise IntegrityError(
            f"{url}: expected {expected_bytes} bytes, received {actual}"
        )


def ensure_component(
    entry: ComponentEntry,
    *,
    fetch: Callable[[str, Path, int], None] | None = None,
) -> Path:
    """Return the local directory holding `entry`'s verified artifacts,
    downloading whatever is missing.

    `fetch` is injectable for tests; the default downloads over HTTPS.
    """
    fetch = fetch or _download
    target_dir = component_dir(entry)
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

"""Registry configuration: alternate index URL and fetch authentication.

Weights stage in private repositories before any public flip, so the fetch
path must work authenticated and then identically unauthenticated. This is a
general alternate-registry capability: any HTTPS index URL, any bearer
token — nothing here is specific to one hosting provider.

Precedence, highest first:

1. Environment: ``CHARACTER_FACTORY_REGISTRY_URL``,
   ``CHARACTER_FACTORY_AUTH_TOKEN``.
2. The config file ``config.json`` in the cache root (see
   :func:`character_factory.registry.store.cache_dir`), keys
   ``registry_url`` and ``auth_token``.
3. Defaults: the vendored snapshot index; anonymous fetches.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from character_factory.registry.store import cache_dir

__all__ = ["RegistryConfig", "load_config"]

ENV_REGISTRY_URL = "CHARACTER_FACTORY_REGISTRY_URL"
ENV_AUTH_TOKEN = "CHARACTER_FACTORY_AUTH_TOKEN"


@dataclass(frozen=True)
class RegistryConfig:
    registry_url: str | None = None
    auth_token: str | None = None

    def headers(self) -> dict[str, str]:
        """HTTP headers applied to every registry and artifact fetch."""
        if self.auth_token:
            return {"Authorization": f"Bearer {self.auth_token}"}
        return {}


def load_config() -> RegistryConfig:
    file_values: dict = {}
    path = cache_dir() / "config.json"
    if path.is_file():
        try:
            file_values = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"unreadable registry config {path}: {error}") from error
        if not isinstance(file_values, dict):
            raise ValueError(f"{path} must contain a JSON object")
    return RegistryConfig(
        registry_url=os.environ.get(ENV_REGISTRY_URL)
        or file_values.get("registry_url"),
        auth_token=os.environ.get(ENV_AUTH_TOKEN) or file_values.get("auth_token"),
    )

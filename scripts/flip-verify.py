#!/usr/bin/env python3
"""flip-verify: one command per verifiable step of the v0.1 launch flip.

Companion to docs/RUNBOOK-v0.1-launch.md — the runbook says *when* to run
each step and what a pass means; this script is the verification itself,
so launch day is one command per step instead of remembered incantations.

    scripts/flip-verify.py step1 [--authenticated]   registry artifacts fetch + hash
    scripts/flip-verify.py step2 [--authenticated]   cold-cache resolve + example assemble
    scripts/flip-verify.py step3 [--expect-sha SHA]  anonymous clone + LICENSE/NOTICE
    scripts/flip-verify.py step4                     fresh venv + PyPI install + first run

Run from a checkout of this repository with the package importable
(``pip install -e .`` or ``PYTHONPATH=src``). Steps 1 and 2 default to
**anonymous** fetches — that is the thing being verified on flip day. The
``--authenticated`` flag applies the configured credentials
(``CHARACTER_FACTORY_REGISTRY_URL`` / ``CHARACTER_FACTORY_AUTH_TOKEN``)
instead, which turns the same checks into the *pre-flip* staging
verification the runbook's preconditions call for.

Every step exits 0 on pass and nonzero with a printed diagnostic on
failure. Before the flip, the anonymous forms are EXPECTED to fail — with
the specific failure modes the runbook documents, and no other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/character-factory/character-factory"
PACKAGE = "character-factory"
EXAMPLE = REPO_ROOT / "examples/characters/marathon-runner.char.json"

_CHUNK = 1 << 20


def fail(message: str) -> int:
    print(f"FAIL {message}")
    return 1


def ok(message: str) -> None:
    print(f"ok   {message}")


# --------------------------------------------------------------------------
# step 1: every registry-referenced artifact fetches and hash-verifies
# --------------------------------------------------------------------------

def step1(authenticated: bool) -> int:
    from character_factory.registry import Registry
    from character_factory.registry.store import ComponentNotPublished, resolve_url

    if authenticated:
        from character_factory.registry.config import load_config

        headers = load_config().headers()
        if not headers:
            return fail("--authenticated but no auth token is configured")
    else:
        headers = {}

    registry = Registry.from_path(
        REPO_ROOT / "src/character_factory/registry/data/registry-snapshot.json"
    )
    checked = 0
    failures = 0
    for entry in registry.index.entries:
        if not entry.artifacts:
            print(f"     {entry.ref}: no artifacts declared (unpublished entry)")
            continue
        for artifact in entry.artifacts:
            try:
                url = resolve_url(entry, artifact["path"])
            except ComponentNotPublished as error:
                failures += 1
                print(f"FAIL {entry.ref}/{artifact['path']}: {error}")
                continue
            request = urllib.request.Request(
                url, headers={"User-Agent": "flip-verify", **headers}
            )
            digest = hashlib.sha256()
            received = 0
            try:
                with urllib.request.urlopen(request) as response:
                    while chunk := response.read(_CHUNK):
                        digest.update(chunk)
                        received += len(chunk)
            except Exception as error:  # noqa: BLE001 — every cause is a finding
                failures += 1
                print(f"FAIL {entry.ref}/{artifact['path']}: fetch failed: {error}")
                continue
            if received != artifact["bytes"]:
                failures += 1
                print(f"FAIL {entry.ref}/{artifact['path']}: "
                      f"{received} bytes, registry pins {artifact['bytes']}")
            elif digest.hexdigest() != artifact["sha256"]:
                failures += 1
                print(f"FAIL {entry.ref}/{artifact['path']}: sha256 mismatch")
            else:
                checked += 1
                ok(f"{entry.ref}/{artifact['path']} "
                   f"({received} bytes, sha256 verified)")
    if failures:
        return fail(f"{failures} artifact(s) failed")
    if not checked:
        return fail("the snapshot declares no fetchable artifacts at all — "
                    "pre-flip state (entries publish with source + artifact "
                    "lists before this step can pass)")
    ok(f"all {checked} declared artifacts fetch and verify "
       f"({'authenticated' if authenticated else 'anonymous'})")
    return 0


# --------------------------------------------------------------------------
# step 2: cold-cache resolution + an example assembles end to end
# --------------------------------------------------------------------------

def step2(authenticated: bool) -> int:
    home = tempfile.mkdtemp(prefix="flip-verify-home-")
    os.environ["CHARACTER_FACTORY_HOME"] = home
    if not authenticated:
        os.environ.pop("CHARACTER_FACTORY_REGISTRY_URL", None)
        os.environ.pop("CHARACTER_FACTORY_AUTH_TOKEN", None)
    print(f"     cold cache: {home} "
          f"({'authenticated' if authenticated else 'anonymous'})")

    from character_factory.registry import Registry, RegistryError

    try:
        if os.environ.get("CHARACTER_FACTORY_REGISTRY_URL"):
            registry = Registry.refresh()
            ok("alternate index refreshed")
        else:
            registry = Registry.default()
            ok("packaged snapshot index")
        for name in ("body-rig", "assembly-assets"):
            path = registry.ensure(name)
            ok(f"{name} resolved and verified into {path}")
    except RegistryError as error:
        return fail(f"cold-cache resolution: {error}")

    from PIL import Image

    from character_factory.api import assemble

    out_dir = Path(tempfile.mkdtemp(prefix="flip-verify-out-"))
    assets = out_dir / "assets"
    assets.mkdir()
    document = json.loads(EXAMPLE.read_text())
    tones = {"skin": (168, 126, 102), "eye": (90, 70, 60),
             "garment": (60, 70, 90), "shoe": (40, 40, 44)}
    for slot in document["textures"]:
        Image.new("RGB", (1024, 1024), tones.get(slot, (128, 128, 128))).save(
            assets / f"{slot}.png"
        )
    try:
        glb = assemble(EXAMPLE, assets, out_dir / "scene.glb", device="cpu")
    except Exception as error:  # noqa: BLE001 — every cause is a finding
        return fail(f"assemble: {error}")
    data = glb.read_bytes()
    if data[:4] != b"glTF" or len(data) < 1 << 20:
        return fail(f"assembled GLB looks wrong ({len(data)} bytes)")
    ok(f"{EXAMPLE.name} assembled from a cold cache: {glb} ({len(data)} bytes)")
    return 0


# --------------------------------------------------------------------------
# step 3: anonymous clone + license files + expected sha
# --------------------------------------------------------------------------

def step3(expect_sha: str | None) -> int:
    clone = Path(tempfile.mkdtemp(prefix="flip-verify-clone-")) / "repo"
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="/bin/false")
    result = subprocess.run(
        ["git", "-c", "credential.helper=", "clone", "--depth", "1",
         REPO_URL, str(clone)],
        env=env, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        return fail(f"anonymous clone refused: {result.stderr.strip().splitlines()[-1]}")
    ok("anonymous clone succeeds")
    for name in ("LICENSE", "NOTICE"):
        if not (clone / name).is_file():
            return fail(f"{name} missing from the public tree")
        ok(f"{name} present")
    head = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"     public HEAD: {head}")
    if expect_sha and not head.startswith(expect_sha):
        return fail(f"public HEAD is not the flip sha {expect_sha}")
    if expect_sha:
        ok("public HEAD matches the flip sha")
    return 0


# --------------------------------------------------------------------------
# step 4: fresh venv, PyPI install, the first-five-minutes path
# --------------------------------------------------------------------------

def step4() -> int:
    root = Path(tempfile.mkdtemp(prefix="flip-verify-venv-"))
    print(f"     fresh venv: {root}")
    venv.EnvBuilder(with_pip=True).create(root)
    pip = root / "bin/pip"
    cli = root / "bin/character-factory"
    # A PYTHONPATH pointing into a checkout makes pip report the package
    # "already satisfied" without installing anything — scrub it so the
    # venv sees only what PyPI actually serves.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    install = subprocess.run(
        [str(pip), "install", "--quiet", PACKAGE],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    if install.returncode != 0:
        return fail("pip install failed: "
                    + install.stderr.strip().splitlines()[-1])
    ok(f"pip install {PACKAGE} succeeds")
    validate = subprocess.run(
        [str(cli), "validate", str(EXAMPLE)],
        env=env, capture_output=True, text=True,
    )
    if validate.returncode != 0:
        return fail(f"validate: {validate.stdout}{validate.stderr}")
    ok(f"character-factory validate passes on {EXAMPLE.name}")
    preflight = subprocess.run(
        [str(cli), "preflight", "--device", "cpu"],
        env=env, capture_output=True, text=True,
    )
    # The base install has no [generation] extra: preflight must exist and
    # must NAME that, not crash.
    if "missing-dependency" not in preflight.stdout and preflight.returncode != 0:
        return fail(f"preflight behaves unexpectedly: {preflight.stdout}"
                    f"{preflight.stderr}")
    ok("character-factory preflight runs and names its causes")
    # The installed package must resolve public weights and assemble — the
    # same check as step2, executed by the venv's interpreter.
    smoke = subprocess.run(
        [str(root / "bin/python"), str(Path(__file__).resolve()), "step2"],
        env=env, capture_output=True, text=True, timeout=3600,
    )
    sys.stdout.write(smoke.stdout)
    if smoke.returncode != 0:
        return fail("installed-package assemble smoke failed (above)")
    ok("installed package assembles from a cold cache")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    s1 = sub.add_parser("step1", help="registry artifacts fetch + hash-verify")
    s1.add_argument("--authenticated", action="store_true")
    s2 = sub.add_parser("step2", help="cold-cache resolution + example assemble")
    s2.add_argument("--authenticated", action="store_true")
    s3 = sub.add_parser("step3", help="anonymous clone + LICENSE/NOTICE + sha")
    s3.add_argument("--expect-sha")
    sub.add_parser("step4", help="fresh venv + PyPI install + first run")
    args = parser.parse_args(argv)
    if args.step == "step1":
        return step1(args.authenticated)
    if args.step == "step2":
        return step2(args.authenticated)
    if args.step == "step3":
        return step3(args.expect_sha)
    return step4()


if __name__ == "__main__":
    raise SystemExit(main())

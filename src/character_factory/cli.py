"""The character-factory command line.

Implemented today: `validate`, `assemble`, `interpret`, `serve`, `mcp`.
`create`/`bake`/`make` land with their modules and are absent, not
stubbed: an unknown command is argparse's honest error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_validate(args: argparse.Namespace) -> int:
    from character_factory.schema import Character, CharacterError

    failures = 0
    for path in args.files:
        try:
            character = Character.load(path, strict=args.strict)
        except CharacterError as error:
            failures += 1
            print(f"FAIL {path}")
            for issue in error.report.errors:
                print(f"     {issue}")
            continue
        except (OSError, ValueError) as error:
            failures += 1
            print(f"FAIL {path}: {error}")
            continue
        label = f"ok   {path}  ({character.content_id[:16]})"
        print(label)
        for warning in character.load_report.warnings:
            print(f"     warning: {warning}")
    return 1 if failures else 0


def _cmd_assemble(args: argparse.Namespace) -> int:
    from character_factory.api import AssetError, assemble
    from character_factory.registry import RegistryError

    try:
        out = assemble(args.character, args.assets, args.output, device=args.device)
    except (AssetError, RegistryError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="character-factory",
        description="Turn a text description into a rigged, textured 3D human.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate", help="validate character files against the format spec"
    )
    validate.add_argument("files", nargs="+", type=Path)
    validate.add_argument(
        "--strict", action="store_true",
        help="treat unknown optional fields as errors",
    )
    validate.set_defaults(func=_cmd_validate)

    assemble = commands.add_parser(
        "assemble", help="build the rigged .glb for a character from baked assets"
    )
    assemble.add_argument("character", type=Path, help="a .char.json file")
    assemble.add_argument(
        "--assets", type=Path, required=True,
        help="directory holding <slot>.png files",
    )
    assemble.add_argument("-o", "--output", type=Path, required=True)
    assemble.add_argument("--device", default="cpu")
    assemble.set_defaults(func=_cmd_assemble)

    interpret = commands.add_parser(
        "interpret",
        help="decompose a description into slot prompts + hair "
             "(prints JSON with wall time and peak memory — the model bench)",
    )
    interpret.add_argument("text", help="the character description")
    interpret.add_argument(
        "--model",
        help="override the configured model: a registry component id or a "
             "local weights path (also CHARACTER_FACTORY_INTERPRETER_MODEL)",
    )
    interpret.add_argument("--device", default="cuda")
    interpret.add_argument(
        "--backend",
        help="select a configured backend by alias (see the server's "
             "/v0/interpreters); 'rules' forces the fallback",
    )
    interpret.add_argument(
        "--rules", action="store_true",
        help="force the deterministic rules fallback (no model)",
    )
    interpret.set_defaults(func=_cmd_interpret)

    preflight = commands.add_parser(
        "preflight",
        help="check the generation stack (imports, CUDA build, driver) "
             "in seconds, with named causes — before any model loads",
    )
    preflight.add_argument("--device", default="cuda")
    preflight.set_defaults(func=_cmd_preflight)

    serve = commands.add_parser(
        "serve", help="run the local /v0 HTTP server (install extra [server])"
    )
    serve.add_argument("--library", type=Path, default=Path("characters"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8400)
    serve.add_argument(
        "--cors-origin", action="append", default=[],
        help="allow a browser origin (repeat for more than one); omitted "
             "means same-origin only",
    )
    serve.set_defaults(func=_cmd_serve)

    mcp = commands.add_parser(
        "mcp", help="run the MCP server on stdio (install extra [mcp])"
    )
    mcp.add_argument("--library", type=Path, default=Path("characters"))
    mcp.set_defaults(func=_cmd_mcp)

    return parser


def _cmd_interpret(args: argparse.Namespace) -> int:
    import dataclasses
    import json

    from character_factory.interpreter import interpret
    from character_factory.interpreter.config import (
        InterpreterConfig,
        load_interpreter_config,
    )

    if args.rules:
        config = InterpreterConfig()          # nothing configured → rules mode
    else:
        config = load_interpreter_config(alias=args.backend)
        if args.model:
            config = dataclasses.replace(config, model=args.model, endpoint=None)
    interpretation, metrics = interpret(
        args.text, device=args.device, config=config
    )
    print(json.dumps(
        {
            "backend": interpretation.backend,
            "textures": {
                slot: {"prompt": prompt}
                for slot, prompt in interpretation.slot_prompts.items()
            },
            "hair": interpretation.hair,
            "notes": interpretation.notes,
            "metrics": metrics,
        },
        indent=2,
    ))
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    from character_factory.preflight import check_generation_stack

    checks = check_generation_stack(device=args.device)
    for check in checks:
        status = "ok  " if check.ok else "FAIL"
        print(f"{status} [{check.name}] {check.detail}")
    return 0 if all(check.ok for check in checks) else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        from character_factory.server import serve
    except ImportError as error:
        print(f"error: the [server] extra is not installed ({error})", file=sys.stderr)
        return 1
    serve(
        args.library, host=args.host, port=args.port,
        cors_origins=args.cors_origin or None,
    )
    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    try:
        from character_factory.mcp import run
    except ImportError as error:
        print(f"error: the [mcp] extra is not installed ({error})", file=sys.stderr)
        return 1
    run(args.library)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Entry point for ZebraLogic."""

from __future__ import annotations

import argparse

from zebralogic import __version__


def greet(name: str = "world") -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}! Welcome to ZebraLogic v{__version__}."


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(prog="zebralogic", description="ZebraLogic CLI.")
    parser.add_argument("--name", default="world", help="who to greet")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    print(greet(args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

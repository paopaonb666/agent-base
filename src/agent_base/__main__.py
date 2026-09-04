"""Enable ``python -m agent_base`` to launch the CLI."""

from agent_base.entrypoints.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

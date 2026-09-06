"""支持通过 ``python -m agent_base`` 启动 CLI。"""

from agent_base.entrypoints.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

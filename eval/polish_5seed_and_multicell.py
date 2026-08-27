"""Backward-compatible entry point for the corrected Load-D reproduction.

Use ``python -m eval.reproduce_main`` in new workflows.
"""

from eval.reproduce_main import main


if __name__ == "__main__":
    raise SystemExit(main())

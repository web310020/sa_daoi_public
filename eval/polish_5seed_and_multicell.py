"""保留旧调用方式的 compatibility entry point，实际转到 corrected Load-D reproduction。

新使用者请运行 ``python -m eval.reproduce_main``。
"""

from eval.reproduce_main import main


if __name__ == "__main__":
    raise SystemExit(main())

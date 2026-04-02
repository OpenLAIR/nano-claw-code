#!/usr/bin/env python3
"""Load project .env (same rules as nano_claw_code.config.load_dotenv) and run pytest.

Usage (from nano-claw-code/):
  python scripts/run_pytest_with_dotenv.py
  python scripts/run_pytest_with_dotenv.py tests/test_tools_impl.py -q

Default pytest addopts skip integration/e2e (see pyproject.toml). To run them:
  python scripts/run_pytest_with_dotenv.py --override-ini addopts= -m "integration or e2e"
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from nano_claw_code.config import load_dotenv

    for k, v in load_dotenv().items():
        if v is not None and k:
            os.environ[k] = str(v)
    cmd = [sys.executable, "-m", "pytest", str(ROOT / "tests"), *sys.argv[1:]]
    if len(sys.argv) == 1:
        cmd.extend(["-v", "--tb=short"])
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

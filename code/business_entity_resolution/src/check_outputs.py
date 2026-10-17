"""Pre-submission check: no CR bytes in the output TSVs, then the competition's validator.

The validator (utils/validate_submission.py) came with the competition kit and is not part
of this repository.
"""

from __future__ import annotations

import subprocess
import sys

from common import DATA, OUT, ROOT


def main() -> None:
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        with open(OUT / name, "rb") as f:
            while chunk := f.read(1 << 24):
                if b"\r" in chunk:
                    sys.exit(f"FAIL: CR byte in {name}")
    r = subprocess.run([sys.executable, str(ROOT / "utils" / "validate_submission.py"),
                        "--matching", str(OUT / "matching_results.tsv"),
                        "--candidate", str(OUT / "candidate_pairs.tsv"),
                        "--test-dir", str(DATA / "test"), "--check-ids"])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()

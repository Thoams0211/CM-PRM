#!/usr/bin/env python3
"""Check whether all samples in a JSON file already contain `answer`.

Exit codes:
- 0: all samples have `answer` (can terminate loop)
- 1: not all samples have `answer`
- 2: invalid file / invalid format
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List


def load_json_as_list(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    data = json.loads(content)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("JSON content must be a list or dict")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether all samples contain answer field")
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the generator/discriminator json file",
    )
    args = parser.parse_args()

    try:
        samples = load_json_as_list(args.path)
    except Exception as e:
        print(f"[answer_check] error: {e}")
        sys.exit(2)

    if not samples:
        print("[answer_check] file is empty, continue loop")
        sys.exit(1)

    total = len(samples)
    done = sum(1 for s in samples if isinstance(s, dict) and "answer" in s)

    print(f"[answer_check] answered {done}/{total}")

    if done == total:
        print("[answer_check] all samples have answer, should stop")
        sys.exit(0)

    print("[answer_check] not finished, continue")
    sys.exit(1)


if __name__ == "__main__":
    main()

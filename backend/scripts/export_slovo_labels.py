#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export labels.txt from Slovo constants.py")
    parser.add_argument("--constants", required=True, help="Path to slovo/constants.py")
    parser.add_argument("--out", required=True, help="Output labels.txt path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    constants_path = Path(args.constants)
    if not constants_path.exists():
        raise FileNotFoundError(f"constants.py not found: {constants_path}")

    src = constants_path.read_text(encoding="utf-8")
    module = ast.parse(src)

    classes: dict[int, str] | None = None
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "classes":
                value = ast.literal_eval(node.value)
                if not isinstance(value, dict):
                    raise ValueError("`classes` in constants.py must be a dict")
                classes = {int(k): str(v) for k, v in value.items()}
                break

    if classes is None:
        raise ValueError("Cannot find `classes = {...}` in constants.py")

    keys = sorted(classes.keys())
    expected = list(range(len(keys)))
    if keys != expected:
        raise ValueError("Classes keys are not contiguous [0..N-1], cannot export deterministic labels")

    labels = [classes[idx] for idx in keys]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

    no_event_label = labels[-1] if labels else ""
    print("[export_slovo_labels] done")
    print(f"  out={out_path}")
    print(f"  labels={len(labels)}")
    print(f"  suggested_no_event_label={no_event_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


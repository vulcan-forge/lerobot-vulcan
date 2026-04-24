#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import shlex
import subprocess
from pathlib import Path

DEFAULT_PARENTS = [
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/sourccey-013__shirt-fold-blue-a/nickm",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/sourccey-013__shirt-fold-blue-c/nickm",
]
HF_LEROBOT_HOME = Path("/home/sourccey/.cache/huggingface/lerobot")
# Default output dataset path under HF_LEROBOT_HOME.
DEFAULT_DATASET_REPO = "Combination/sourccey-shirt-fold-c-001"


def is_dataset_root(path: Path) -> bool:
    return (
        (path / "meta" / "info.json").is_file()
        and (path / "meta" / "tasks.parquet").is_file()
        and (path / "data").is_dir()
    )


def discover_dataset_roots(parents: list[Path]) -> list[Path]:
    roots: list[Path] = []

    for parent in parents:
        if not parent.exists():
            raise SystemExit(f"Missing parent folder: {parent}")

        if is_dataset_root(parent):
            roots.append(parent.resolve())
            continue

        for child in sorted(parent.iterdir()):
            if child.is_dir() and is_dataset_root(child):
                roots.append(child.resolve())

    deduped = sorted(dict.fromkeys(roots))
    return deduped


def default_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge all LeRobot datasets found under one or more parent directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parent",
        dest="parents",
        action="append",
        default=[],
        help="Parent folder containing dataset roots. Can be repeated.",
    )
    parser.add_argument(
        "--out-repo",
        dest="out_repo",
        default=None,
        help="Merged output repo id (e.g. Combination/my-merged-dataset).",
    )
    parser.add_argument(
        "--out-root",
        dest="out_root",
        default=None,
        help="Merged output dataset root path. Defaults to $HF_LEROBOT_HOME/<out-repo>.",
    )
    parser.add_argument(
        "--config-path",
        dest="config_path",
        default=None,
        help="Path to write merge config json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build config and print details without running lerobot-edit-dataset.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list discovered dataset roots and exit.",
    )
    parser.add_argument(
        "--verbose-roots",
        action="store_true",
        help="Print every discovered dataset root.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    parents_raw = args.parents if args.parents else DEFAULT_PARENTS
    parents = [Path(p).expanduser() for p in parents_raw]

    stamp = default_stamp()
    out_repo = args.out_repo or DEFAULT_DATASET_REPO
    out_root = Path(args.out_root).expanduser() if args.out_root else (HF_LEROBOT_HOME / out_repo)
    config_path = (
        Path(args.config_path).expanduser()
        if args.config_path
        else Path(f"/tmp/sourccey_merge_blue_ac_{stamp}.json")
    )

    roots = discover_dataset_roots(parents)
    if len(roots) < 2:
        raise SystemExit(f"Need at least 2 dataset roots to merge, found {len(roots)}")

    print(f"Found {len(roots)} dataset roots")
    if args.verbose_roots or args.list_only:
        for root in roots:
            print(f"  - {root}")
    else:
        preview_count = min(10, len(roots))
        print("Preview:")
        for root in roots[:preview_count]:
            print(f"  - {root}")
        if len(roots) > preview_count:
            print(f"  ... ({len(roots) - preview_count} more; use --verbose-roots to print all)")

    if args.list_only:
        return 0

    repo_ids = [root.name for root in roots]
    cfg = {
        "new_repo_id": out_repo,
        "new_root": str(out_root),
        "operation": {
            "type": "merge",
            "repo_ids": repo_ids,
            "roots": [str(root) for root in roots],
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2))

    print(f"\nWrote merge config: {config_path}")
    print(f"Output repo_id: {out_repo}")
    print(f"Output root: {out_root}")

    cmd = ["uv", "run", "lerobot-edit-dataset", "--config_path", str(config_path)]
    print(f"\nRunning: {' '.join(shlex.quote(x) for x in cmd)}")

    if args.dry_run:
        print("Dry run enabled; not executing merge command.")
        return 0

    subprocess.run(cmd, check=True)

    print("\nMerge completed.")
    print(f"Merged dataset written to: {out_root}")
    print("\nNext command:")
    print(
        "uv run lerobot-edit-dataset "
        f"--repo_id '{out_repo}' --root '{out_root}' --operation.type info --operation.show_features true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

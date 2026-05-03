#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import shlex
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PARENTS = [
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/sourccey-013__shirt-fold-blue-a/nickm",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/sourccey-013__shirt-fold-blue-c/nickm",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/rollout__sourccey-013__shirt-fold-blue-c/nickm",
]
HF_LEROBOT_HOME = Path("/home/sourccey/.cache/huggingface/lerobot")
DEFAULT_DATASET_REPO = "Combination/sourccey-shirt-fold-c-004"


@dataclass(frozen=True)
class DatasetCandidate:
    root: Path
    repo_id: str
    features: dict


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

    return sorted(dict.fromkeys(roots))


def load_features(path: Path) -> dict:
    info_path = path / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing info.json for dataset root: {path}") from exc

    features = info.get("features")
    if not isinstance(features, dict):
        raise SystemExit(f"Invalid features metadata in {info_path}")
    return features


def feature_signature(features: dict) -> str:
    return json.dumps(features, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_candidates(roots: list[Path]) -> list[DatasetCandidate]:
    return [DatasetCandidate(root=root, repo_id=root.name, features=load_features(root)) for root in roots]


def filter_candidates_by_feature(
    candidates: list[DatasetCandidate], require_features: list[str], exclude_features: list[str]
) -> tuple[list[DatasetCandidate], list[DatasetCandidate]]:
    if not require_features and not exclude_features:
        return candidates, []

    kept: list[DatasetCandidate] = []
    dropped: list[DatasetCandidate] = []
    require_set = set(require_features)
    exclude_set = set(exclude_features)

    for candidate in candidates:
        feature_keys = set(candidate.features.keys())
        if require_set and not require_set.issubset(feature_keys):
            dropped.append(candidate)
            continue
        if exclude_set and (exclude_set & feature_keys):
            dropped.append(candidate)
            continue
        kept.append(candidate)

    return kept, dropped


def group_candidates_by_schema(
    candidates: list[DatasetCandidate],
) -> tuple[dict[str, list[DatasetCandidate]], dict[str, dict]]:
    grouped: dict[str, list[DatasetCandidate]] = defaultdict(list)
    features_by_sig: dict[str, dict] = {}

    for candidate in candidates:
        sig = feature_signature(candidate.features)
        grouped[sig].append(candidate)
        if sig not in features_by_sig:
            features_by_sig[sig] = candidate.features

    for sig in grouped:
        grouped[sig].sort(key=lambda c: str(c.root))

    return dict(grouped), features_by_sig


def schema_label(idx: int) -> str:
    return f"schema_{idx:02d}"


def print_root_preview(candidates: list[DatasetCandidate], verbose: bool) -> None:
    if verbose:
        for candidate in candidates:
            print(f"  - {candidate.root}")
        return

    preview_count = min(10, len(candidates))
    print("Preview:")
    for candidate in candidates[:preview_count]:
        print(f"  - {candidate.root}")
    if len(candidates) > preview_count:
        print(f"  ... ({len(candidates) - preview_count} more; use --verbose-roots to print all)")


def print_schema_groups(grouped: dict[str, list[DatasetCandidate]], features_by_sig: dict[str, dict]) -> None:
    print("\nDetected multiple feature schemas:")
    for idx, (sig, candidates) in enumerate(
        sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True), start=1
    ):
        features = features_by_sig[sig]
        feature_keys = sorted(features.keys())
        print(f"  [{idx}] {schema_label(idx)}: {len(candidates)} datasets")
        print(f"      features: {', '.join(feature_keys)}")
        print(f"      sample: {candidates[0].root}")


def resolve_group_root(base_out_root: Path | None, repo_id: str) -> Path:
    if base_out_root is None:
        return HF_LEROBOT_HOME / repo_id
    return base_out_root.parent / repo_id


def run_merge(
    candidates: list[DatasetCandidate], out_repo: str, out_root: Path, config_path: Path, dry_run: bool
) -> None:
    repo_ids = [candidate.repo_id for candidate in candidates]
    cfg = {
        "new_repo_id": out_repo,
        "new_root": str(out_root),
        "operation": {
            "type": "merge",
            "repo_ids": repo_ids,
            "roots": [str(candidate.root) for candidate in candidates],
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2))

    print(f"\nWrote merge config: {config_path}")
    print(f"Output repo_id: {out_repo}")
    print(f"Output root: {out_root}")

    cmd = ["uv", "run", "lerobot-edit-dataset", "--config_path", str(config_path)]
    print(f"\nRunning: {' '.join(shlex.quote(x) for x in cmd)}")

    if dry_run:
        print("Dry run enabled; not executing merge command.")
        return

    subprocess.run(cmd, check=True)


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
    parser.add_argument(
        "--require-feature",
        action="append",
        default=[],
        help="Keep only datasets that contain this feature. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        help="Drop datasets that contain this feature. Can be repeated.",
    )
    parser.add_argument(
        "--split-by-features",
        action="store_true",
        help="If mixed schemas are found, create one merged output per schema instead of failing.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    parents_raw = args.parents if args.parents else DEFAULT_PARENTS
    parents = [Path(p).expanduser() for p in parents_raw]

    stamp = default_stamp()
    out_repo = args.out_repo or DEFAULT_DATASET_REPO
    base_out_root = Path(args.out_root).expanduser() if args.out_root else None
    out_root = base_out_root if base_out_root else (HF_LEROBOT_HOME / out_repo)
    config_path = (
        Path(args.config_path).expanduser()
        if args.config_path
        else Path(f"/tmp/sourccey_merge_blue_ac_{stamp}.json")
    )

    roots = discover_dataset_roots(parents)
    candidates = build_candidates(roots)

    effective_exclusions = sorted(set(args.exclude_feature))
    candidates, dropped = filter_candidates_by_feature(candidates, args.require_feature, effective_exclusions)

    if dropped:
        print(f"Dropped {len(dropped)} roots due to feature filters")

    if len(candidates) < 2:
        raise SystemExit(f"Need at least 2 dataset roots to merge, found {len(candidates)}")

    print(f"Found {len(candidates)} dataset roots")
    print_root_preview(candidates, verbose=args.verbose_roots or args.list_only)

    grouped, features_by_sig = group_candidates_by_schema(candidates)
    if args.list_only:
        if len(grouped) > 1:
            print_schema_groups(grouped, features_by_sig)
        return 0

    if len(grouped) > 1:
        print_schema_groups(grouped, features_by_sig)
        if not args.split_by_features:
            raise SystemExit(
                "\nMixed feature schemas cannot be merged in one pass.\n"
                "Fix options:\n"
                "  1) Narrow by schema with --require-feature / --exclude-feature\n"
                "  2) Auto-merge each schema separately: --split-by-features\n"
            )

        used_labels: set[str] = set()
        merge_jobs: list[tuple[str, list[DatasetCandidate], Path, Path]] = []
        for idx, (_, schema_candidates) in enumerate(
            sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True), start=1
        ):
            label_base = schema_label(idx)
            label = label_base
            suffix = 2
            while label in used_labels:
                label = f"{label_base}_{suffix}"
                suffix += 1
            used_labels.add(label)

            group_repo = f"{out_repo}__{label}"
            group_root = resolve_group_root(base_out_root, group_repo)
            group_config = config_path.with_name(f"{config_path.stem}__{label}{config_path.suffix}")
            merge_jobs.append((group_repo, schema_candidates, group_root, group_config))

        print("\nPlanned schema-specific merges:")
        for repo, schema_candidates, group_root, _ in merge_jobs:
            print(f"  - {repo}: {len(schema_candidates)} datasets -> {group_root}")

        for repo, schema_candidates, group_root, group_config in merge_jobs:
            run_merge(schema_candidates, repo, group_root, group_config, dry_run=args.dry_run)

        print("\nSchema-split merge completed.")
        print("Use one of these depending on training target:")
        for repo, _, group_root, _ in merge_jobs:
            print(f"  - repo_id={repo} root={group_root}")
        return 0

    run_merge(candidates, out_repo, out_root, config_path, dry_run=args.dry_run)

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

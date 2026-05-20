#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import json
import shlex
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PARENTS = [
    "/home/sourccey/.cache/huggingface/lerobot/Combination/sourccey-shirt-fold-c-006",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-blue-c-specific/sourccey-013__shirt-fold-blue-c-specific-005",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-blue-c-specific/sourccey-013__shirt-fold-blue-c-specific-006",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-blue-c-specific/sourccey-013__shirt-fold-blue-c-specific-007",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-blue-c-specific/sourccey-013__shirt-fold-blue-c-specific-008",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-blue-c-specific/sourccey-013__shirt-fold-blue-c-specific-009",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-blue-c-specific/sourccey-013__shirt-fold-blue-c-specific-010",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-light-blue-c/sourccey-013__shirt-fold-light-blue-c__set000_nickm",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-light-blue-c/sourccey-013__shirt-fold-light-blue-c__set001_nickm",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-light-blue-c/sourccey-013__shirt-fold-light-blue-c__set002_nickm",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-light-blue-c/sourccey-013__shirt-fold-light-blue-c__set003_nickm",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-light-blue-c/sourccey-013__shirt-fold-light-blue-c__set004_nickm",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-light-blue-c/sourccey-013__shirt-fold-light-blue-c__set005_nickm",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-light-blue-c/sourccey-013__shirt-fold-light-blue-c__set006_nickm",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-light-blue-c/sourccey-013__shirt-fold-light-blue-c__set007_nickm",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-light-blue-c/sourccey-013__shirt-fold-light-blue-c__set008_nickm",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-light-blue-c/sourccey-013__shirt-fold-light-blue-c__set009_nickm",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/nickm/sourccey-013__shirt-fold-light-blue-c/sourccey-013__shirt-fold-light-blue-c__set010_nickm",
]
HF_LEROBOT_HOME = Path("/home/sourccey/.cache/huggingface/lerobot")
DEFAULT_DATASET_REPO = "Combination/sourccey-shirt-fold-c-008"


@dataclass(frozen=True)
class DatasetCandidate:
    root: Path
    repo_id: str
    features: dict
    total_episodes: int


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


def load_dataset_info(path: Path) -> tuple[dict, int]:
    info_path = path / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing info.json for dataset root: {path}") from exc

    features = info.get("features")
    if not isinstance(features, dict):
        raise SystemExit(f"Invalid features metadata in {info_path}")

    total_episodes = info.get("total_episodes")
    if not isinstance(total_episodes, int) or total_episodes < 0:
        raise SystemExit(
            f"Invalid total_episodes in {info_path}: expected non-negative int, got {total_episodes!r}"
        )

    return features, total_episodes


def feature_signature(features: dict) -> str:
    return json.dumps(features, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_candidates(roots: list[Path]) -> list[DatasetCandidate]:
    candidates: list[DatasetCandidate] = []
    for root in roots:
        features, total_episodes = load_dataset_info(root)
        candidates.append(
            DatasetCandidate(
                root=root,
                repo_id=root.name,
                features=features,
                total_episodes=total_episodes,
            )
        )
    return candidates


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


def lineage_csv_path(out_root: Path) -> Path:
    return out_root / "_reports" / "episode_lineage.csv"


def write_episode_lineage_csv(candidates: list[DatasetCandidate], csv_path: Path) -> tuple[Path, int]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    merged_episode_index = 0
    expected_row_count = sum(candidate.total_episodes for candidate in candidates)
    fields = [
        "merged_episode_index",
        "source_dataset_index",
        "source_repo_id",
        "source_root",
        "source_episode_index",
    ]

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source_dataset_index, candidate in enumerate(candidates):
            for source_episode_index in range(candidate.total_episodes):
                writer.writerow(
                    {
                        "merged_episode_index": merged_episode_index,
                        "source_dataset_index": source_dataset_index,
                        "source_repo_id": candidate.repo_id,
                        "source_root": str(candidate.root),
                        "source_episode_index": source_episode_index,
                    }
                )
                merged_episode_index += 1
                row_count += 1

    if row_count != expected_row_count:
        raise SystemExit(
            f"Lineage row count mismatch for {csv_path}: wrote {row_count}, expected {expected_row_count}"
        )

    return csv_path, row_count


def run_merge(
    candidates: list[DatasetCandidate], out_repo: str, out_root: Path, config_path: Path, dry_run: bool
) -> None:
    final_lineage_path = lineage_csv_path(out_root)
    if dry_run:
        premerge_lineage_path = final_lineage_path
    else:
        if out_root.exists():
            raise SystemExit(
                f"Output root already exists and merge requires a fresh directory: {out_root}\n"
                "If this is from a prior failed run, remove it first and retry."
            )
        premerge_lineage_path = config_path.with_name(f"{config_path.stem}__episode_lineage.csv")

    lineage_path, lineage_rows = write_episode_lineage_csv(candidates, premerge_lineage_path)

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
    print(f"Wrote episode lineage CSV: {lineage_path} ({lineage_rows} rows)")

    cmd = ["uv", "run", "lerobot-edit-dataset", "--config_path", str(config_path)]
    print(f"\nRunning: {' '.join(shlex.quote(x) for x in cmd)}")

    if dry_run:
        print("Dry run enabled; not executing merge command.")
        return

    subprocess.run(cmd, check=True)

    if lineage_path != final_lineage_path:
        final_lineage_path.parent.mkdir(parents=True, exist_ok=True)
        lineage_path.replace(final_lineage_path)
        print(f"Moved episode lineage CSV to: {final_lineage_path}")


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

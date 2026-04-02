#!/usr/bin/env python

import argparse
import json
import logging
import shlex
import shutil
from typing import List

from lerobot.datasets.sourccey_aggregate import (
    aggregate_datasets_full_validation,
    append_to_base_dataset_full_validation,
)
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


def combine_datasets_full_validation(
    dataset_paths: List[str],
    output_path: str,
    auto_repair_source_clips: bool = True,
) -> None:
    """Combine multiple LeRobot datasets with full video decode validation."""
    logging.info(
        "Combining %d datasets into %s with full video decode validation",
        len(dataset_paths),
        output_path,
    )

    aggregate_datasets_full_validation(
        repo_ids=dataset_paths,
        aggr_repo_id=output_path,
        auto_repair_source_clips=auto_repair_source_clips,
    )

    logging.info("Successfully combined datasets into %s", output_path)


def append_datasets_full_validation(
    base_dataset_path: str,
    new_dataset_paths: List[str],
    append_output_path: str | None = None,
    auto_repair_source_clips: bool = True,
) -> None:
    """Append new datasets onto an existing combined dataset.

    If append_output_path is provided, base dataset is first cloned to
    append_output_path, then new datasets are appended there.
    """
    target_dataset_path = base_dataset_path
    if append_output_path is not None:
        if append_output_path == base_dataset_path:
            raise ValueError("--append_output_path must be different from --base_dataset_path.")
        src_root = HF_LEROBOT_HOME / base_dataset_path
        dst_root = HF_LEROBOT_HOME / append_output_path
        if not src_root.exists():
            raise FileNotFoundError(
                f"Base dataset path does not exist under HF cache: {src_root}"
            )
        if dst_root.exists():
            raise FileExistsError(
                f"Append output dataset already exists: {dst_root}. "
                "Choose a new --append_output_path."
            )
        logging.info("Cloning base dataset %s -> %s", base_dataset_path, append_output_path)
        shutil.copytree(src_root, dst_root)
        target_dataset_path = append_output_path

    logging.info(
        "Appending %d new datasets into dataset %s with full validation",
        len(new_dataset_paths),
        target_dataset_path,
    )
    append_to_base_dataset_full_validation(
        base_repo_id=target_dataset_path,
        new_repo_ids=new_dataset_paths,
        auto_repair_source_clips=auto_repair_source_clips,
    )
    logging.info("Successfully appended datasets into %s", target_dataset_path)


def _parse_dataset_paths(dataset_paths_arg: str) -> List[str]:
    """Parse dataset paths from command line argument.

    Supports both JSON array format and shell-style space-separated format.
    """
    if dataset_paths_arg.strip().startswith("[") and dataset_paths_arg.strip().endswith("]"):
        try:
            cleaned_arg = dataset_paths_arg.strip().replace(",]", "]").replace(", ]", "]")
            parsed_paths = json.loads(cleaned_arg)
            return [path for path in parsed_paths if path and isinstance(path, str) and path.strip()]
        except json.JSONDecodeError as exc:
            logging.warning("Failed to parse dataset_paths as JSON array (%s). Falling back to shell parsing.", exc)

    parsed_paths = shlex.split(dataset_paths_arg)
    return [path for path in parsed_paths if path and path.strip() and path not in ["[", "]"]]


def _find_zero_byte_parquet_files(dataset_repo_id: str) -> list[str]:
    dataset_root = HF_LEROBOT_HOME / dataset_repo_id
    if not dataset_root.exists():
        return [f"{dataset_repo_id}: dataset path does not exist under {HF_LEROBOT_HOME}"]

    bad_paths: list[str] = []
    for subdir in ("data", "meta"):
        target = dataset_root / subdir
        if not target.exists():
            continue
        for fpath in target.rglob("*.parquet"):
            try:
                if fpath.stat().st_size == 0:
                    bad_paths.append(str(fpath.relative_to(HF_LEROBOT_HOME)))
            except OSError as exc:
                bad_paths.append(f"{dataset_repo_id}: failed to stat {fpath} ({exc})")
    return bad_paths


def _validate_new_dataset_paths(dataset_paths: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    valid: list[str] = []
    invalid: list[tuple[str, str]] = []

    for repo_id in dataset_paths:
        zero_byte = _find_zero_byte_parquet_files(repo_id)
        if zero_byte:
            invalid.append((repo_id, f"contains invalid parquet files: {zero_byte}"))
            continue

        try:
            # Metadata load catches additional structural issues beyond zero-byte parquet.
            LeRobotDatasetMetadata(repo_id)
            valid.append(repo_id)
        except Exception as exc:  # noqa: BLE001
            invalid.append((repo_id, f"metadata load failed: {exc}"))

    return valid, invalid


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple LeRobot datasets with full validation, or append new datasets "
            "onto an existing verified combined dataset."
        )
    )
    parser.add_argument(
        "--dataset_paths",
        type=str,
        help=(
            "Dataset paths to combine (full combine mode). Supports JSON array format "
            '(e.g. \'["repo1/dataset1", "repo2/dataset2"]\') or shell-style space-separated list.'
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path/repo ID for full combine mode.",
    )
    parser.add_argument(
        "--base_dataset_path",
        type=str,
        help="Base combined dataset path/repo ID for append mode.",
    )
    parser.add_argument(
        "--new_dataset_paths",
        type=str,
        help=(
            "New dataset paths to append in append mode. Supports JSON array format "
            '(e.g. \'["repo1/dataset1", "repo2/dataset2"]\') or shell-style space-separated list.'
        ),
    )
    parser.add_argument(
        "--append_output_path",
        type=str,
        help=(
            "Append mode only. If set, clone --base_dataset_path into this new dataset path "
            "and append there (base remains untouched)."
        ),
    )
    parser.add_argument(
        "--skip_invalid_new_datasets",
        action="store_true",
        help=(
            "Append mode only. Skip new datasets that fail preflight checks "
            "(e.g. zero-byte parquet or metadata load failure) instead of failing."
        ),
    )
    parser.add_argument(
        "--no_auto_repair_source_clips",
        action="store_true",
        help="Disable auto-repair for source clips that fail decode validation.",
    )
    args = parser.parse_args()

    auto_repair = not args.no_auto_repair_source_clips

    # Append mode
    if (
        args.base_dataset_path is not None
        or args.new_dataset_paths is not None
        or args.append_output_path is not None
    ):
        if args.base_dataset_path is None or args.new_dataset_paths is None:
            raise ValueError(
                "Append mode requires both --base_dataset_path and --new_dataset_paths."
            )
        if args.output_path is not None:
            raise ValueError(
                "--output_path is full-combine only. In append mode, use --append_output_path "
                "if you want a new output dataset clone."
            )
        new_dataset_paths = _parse_dataset_paths(args.new_dataset_paths)
        logging.info("Parsed %d new dataset paths for append", len(new_dataset_paths))

        valid_paths, invalid_paths = _validate_new_dataset_paths(new_dataset_paths)
        if invalid_paths:
            preview = "\n".join([f" - {repo_id}: {reason}" for repo_id, reason in invalid_paths[:10]])
            if args.skip_invalid_new_datasets:
                logging.warning(
                    "Skipping %d invalid new datasets due to preflight failures:\n%s",
                    len(invalid_paths),
                    preview,
                )
                new_dataset_paths = valid_paths
            else:
                raise ValueError(
                    "Invalid new datasets found during preflight checks. "
                    "Fix/remove these datasets, or rerun with --skip_invalid_new_datasets.\n"
                    f"{preview}"
                )

        append_datasets_full_validation(
            base_dataset_path=args.base_dataset_path,
            new_dataset_paths=new_dataset_paths,
            append_output_path=args.append_output_path,
            auto_repair_source_clips=auto_repair,
        )
        if invalid_paths and args.skip_invalid_new_datasets:
            skipped_list = "\n".join([f" - {repo_id}: {reason}" for repo_id, reason in invalid_paths])
            logging.warning("Combine completed but these files were not able to be appended:\n%s", skipped_list)
        return

    # Full combine mode
    if args.dataset_paths is None or args.output_path is None:
        raise ValueError(
            "Full combine mode requires --dataset_paths and --output_path "
            "(or use append mode with --base_dataset_path and --new_dataset_paths)."
        )

    dataset_paths = _parse_dataset_paths(args.dataset_paths)
    logging.info("Parsed %d dataset paths", len(dataset_paths))
    combine_datasets_full_validation(
        dataset_paths=dataset_paths,
        output_path=args.output_path,
        auto_repair_source_clips=auto_repair,
    )


if __name__ == "__main__":
    init_logging()
    main()

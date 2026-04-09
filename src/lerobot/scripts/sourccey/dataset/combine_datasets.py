#!/usr/bin/env python

import argparse
import json
import logging
import shlex
import shutil
from typing import List

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.sourccey_aggregate import (
    aggregate_datasets,
    aggregate_datasets_full_validation,
    append_to_base_dataset_full_validation,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


def combine_datasets(
    dataset_paths: List[str],
    output_path: str,
    validate_output: bool = True,
    auto_repair_source_clips: bool = True,
) -> None:
    """Combine multiple LeRobot datasets, validating output by default."""
    if validate_output:
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
    else:
        logging.info(
            "Combining %d datasets into %s without full validation",
            len(dataset_paths),
            output_path,
        )
        aggregate_datasets(repo_ids=dataset_paths, aggr_repo_id=output_path)

    logging.info("Successfully combined datasets into %s", output_path)


def append_datasets(
    base_dataset_path: str,
    new_dataset_paths: List[str],
    append_output_path: str,
    auto_repair_source_clips: bool = True,
) -> None:
    """Clone a verified combined dataset, then append new datasets into the clone."""
    if append_output_path == base_dataset_path:
        raise ValueError("--append_output_path must be different from --base_dataset_path.")

    src_root = HF_LEROBOT_HOME / base_dataset_path
    dst_root = HF_LEROBOT_HOME / append_output_path
    if not src_root.exists():
        raise FileNotFoundError(f"Base dataset path does not exist under HF cache: {src_root}")
    if dst_root.exists():
        raise FileExistsError(
            f"Append output dataset already exists: {dst_root}. "
            "Choose a new --append_output_path."
        )

    logging.info("Cloning base dataset %s -> %s", base_dataset_path, append_output_path)
    shutil.copytree(src_root, dst_root)

    logging.info(
        "Appending %d new datasets into cloned dataset %s with full validation",
        len(new_dataset_paths),
        append_output_path,
    )
    append_to_base_dataset_full_validation(
        base_repo_id=append_output_path,
        new_repo_ids=new_dataset_paths,
        auto_repair_source_clips=auto_repair_source_clips,
    )
    logging.info("Successfully appended datasets into %s", append_output_path)


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
            logging.warning(
                "Failed to parse dataset_paths as JSON array (%s). Falling back to shell parsing.",
                exc,
            )

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


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{value}'. Use true/false."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple LeRobot datasets with validation enabled by default, "
            "or clone a verified combined dataset and append new datasets into the clone."
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
        "--validate_output",
        type=_parse_bool,
        default=True,
        help=(
            "Full combine mode only. Defaults to true. Set to false to skip full validation "
            "and use the older raw combine behavior."
        ),
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
        help="Append mode only. Required. Clone --base_dataset_path into this new dataset path and append there.",
    )
    parser.add_argument(
        "--no_auto_repair_source_clips",
        action="store_true",
        help="Disable auto-repair for source clips that fail decode validation.",
    )
    args = parser.parse_args()

    auto_repair = not args.no_auto_repair_source_clips

    if (
        args.base_dataset_path is not None
        or args.new_dataset_paths is not None
        or args.append_output_path is not None
    ):
        if (
            args.base_dataset_path is None
            or args.new_dataset_paths is None
            or args.append_output_path is None
        ):
            raise ValueError(
                "Append mode requires --base_dataset_path, --new_dataset_paths, and --append_output_path."
            )
        if args.output_path is not None:
            raise ValueError("--output_path is full-combine only.")
        if not args.validate_output:
            raise ValueError(
                "--validate_output=false is only supported for full combine mode. "
                "Append mode always validates."
            )

        new_dataset_paths = _parse_dataset_paths(args.new_dataset_paths)
        logging.info("Parsed %d new dataset paths for append", len(new_dataset_paths))

        valid_paths, invalid_paths = _validate_new_dataset_paths(new_dataset_paths)
        if invalid_paths:
            preview = "\n".join([f" - {repo_id}: {reason}" for repo_id, reason in invalid_paths[:10]])
            logging.warning(
                "Skipping %d invalid new datasets due to preflight failures:\n%s",
                len(invalid_paths),
                preview,
            )
            new_dataset_paths = valid_paths
        if not new_dataset_paths:
            logging.warning("No valid new datasets remain after preflight checks. Nothing to append.")
            return

        append_datasets(
            base_dataset_path=args.base_dataset_path,
            new_dataset_paths=new_dataset_paths,
            append_output_path=args.append_output_path,
            auto_repair_source_clips=auto_repair,
        )
        if invalid_paths:
            skipped_list = "\n".join([f" - {repo_id}: {reason}" for repo_id, reason in invalid_paths])
            logging.warning("Append completed but these datasets were skipped:\n%s", skipped_list)
        return

    if args.dataset_paths is None or args.output_path is None:
        raise ValueError(
            "Full combine mode requires --dataset_paths and --output_path "
            "(or use append mode with --base_dataset_path, --new_dataset_paths, and --append_output_path)."
        )

    dataset_paths = _parse_dataset_paths(args.dataset_paths)
    logging.info("Parsed %d dataset paths", len(dataset_paths))

    invalid_paths: list[tuple[str, str]] = []
    if args.validate_output:
        valid_paths, invalid_paths = _validate_new_dataset_paths(dataset_paths)
        if invalid_paths:
            preview = "\n".join([f" - {repo_id}: {reason}" for repo_id, reason in invalid_paths[:10]])
            logging.warning(
                "Skipping %d invalid datasets due to preflight failures:\n%s",
                len(invalid_paths),
                preview,
            )
            dataset_paths = valid_paths
            if not dataset_paths:
                raise ValueError("All provided datasets failed preflight checks; nothing left to combine.")
    else:
        logging.info(
            "Full validation disabled via --validate_output=false; "
            "running raw combine without preflight checks."
        )

    combine_datasets(
        dataset_paths=dataset_paths,
        output_path=args.output_path,
        validate_output=args.validate_output,
        auto_repair_source_clips=auto_repair,
    )
    if invalid_paths:
        skipped_list = "\n".join([f" - {repo_id}: {reason}" for repo_id, reason in invalid_paths])
        logging.warning("Combine completed but these datasets were skipped:\n%s", skipped_list)


if __name__ == "__main__":
    init_logging()
    main()

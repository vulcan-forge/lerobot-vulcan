#!/usr/bin/env python3
"""Fail if uv.lock pins package artifacts newer than a minimum age.

This script is intended for dependency update workflows that run `uv lock --upgrade`.
It inspects `upload-time` values stored in `uv.lock` (for sdist/wheels) and rejects
package versions whose newest artifact is younger than the configured threshold.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Violation:
    name: str
    version: str
    source: str
    newest_upload: dt.datetime
    age_days: float


@dataclass(frozen=True)
class SourceInfo:
    raw: str
    is_editable_local: bool
    is_pypi: bool


def _parse_iso8601_utc(value: str) -> dt.datetime:
    # uv.lock stores times like "2026-03-04T19:34:12.359Z".
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _collect_upload_times(package: dict[str, Any]) -> list[dt.datetime]:
    out: list[dt.datetime] = []
    sdist = package.get("sdist")
    if isinstance(sdist, dict):
        upload_time = sdist.get("upload-time")
        if isinstance(upload_time, str):
            out.append(_parse_iso8601_utc(upload_time))

    wheels = package.get("wheels")
    if isinstance(wheels, list):
        for wheel in wheels:
            if not isinstance(wheel, dict):
                continue
            upload_time = wheel.get("upload-time")
            if isinstance(upload_time, str):
                out.append(_parse_iso8601_utc(upload_time))
    return out


def _source_info(package: dict[str, Any]) -> SourceInfo:
    source_info = package.get("source")
    if isinstance(source_info, dict):
        if "editable" in source_info:
            return SourceInfo(raw=str(source_info), is_editable_local=True, is_pypi=False)
        registry = source_info.get("registry")
        if isinstance(registry, str):
            return SourceInfo(raw=registry, is_editable_local=False, is_pypi=registry.startswith("https://pypi.org/"))
        return SourceInfo(raw=str(source_info), is_editable_local=False, is_pypi=False)
    return SourceInfo(raw=str(source_info or ""), is_editable_local=False, is_pypi=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce minimum package age in uv.lock.")
    parser.add_argument("--lock-file", default="uv.lock", help="Path to uv.lock (default: uv.lock)")
    parser.add_argument(
        "--min-age-days",
        type=float,
        default=3.0,
        help="Minimum allowed age in days for locked artifacts (default: 3.0)",
    )
    parser.add_argument(
        "--allow-missing-upload-time",
        action="store_true",
        help="Allow packages that have no upload-time metadata (default: fail closed).",
    )
    args = parser.parse_args()

    lock_path = Path(args.lock_file)
    if not lock_path.exists():
        print(f"ERROR: lock file not found: {lock_path}", file=sys.stderr)
        return 2

    with lock_path.open("rb") as f:
        lock_data = tomllib.load(f)

    packages = lock_data.get("package", [])
    if not isinstance(packages, list):
        print("ERROR: invalid uv.lock format: 'package' must be a list", file=sys.stderr)
        return 2

    now = dt.datetime.now(dt.timezone.utc)
    min_age = dt.timedelta(days=max(0.0, args.min_age_days))

    violations: list[Violation] = []
    missing_times_pypi: list[tuple[str, str, str]] = []
    skipped_missing_times_non_pypi: list[tuple[str, str, str]] = []

    for package in packages:
        if not isinstance(package, dict):
            continue
        name = str(package.get("name", "<unknown>"))
        version = str(package.get("version", "<unknown>"))
        source = _source_info(package)

        upload_times = _collect_upload_times(package)
        if not upload_times:
            if source.is_editable_local:
                continue
            if source.is_pypi:
                missing_times_pypi.append((name, version, source.raw))
            else:
                skipped_missing_times_non_pypi.append((name, version, source.raw))
            continue

        newest = max(upload_times)
        age = now - newest
        if age < min_age:
            violations.append(
                Violation(
                    name=name,
                    version=version,
                    source=source.raw,
                    newest_upload=newest,
                    age_days=age.total_seconds() / 86400.0,
                )
            )

    if missing_times_pypi and not args.allow_missing_upload_time:
        print("ERROR: found packages in uv.lock without upload-time metadata:", file=sys.stderr)
        for name, version, source in sorted(missing_times_pypi):
            src = source or "<unknown source>"
            print(f"  - {name}=={version} (source: {src})", file=sys.stderr)
        print(
            "Refusing to continue. Re-run with --allow-missing-upload-time to bypass.",
            file=sys.stderr,
        )
        return 1

    if violations:
        print(
            f"ERROR: {len(violations)} package(s) are newer than the required "
            f"{args.min_age_days:.2f} day(s):",
            file=sys.stderr,
        )
        for v in sorted(violations, key=lambda item: item.age_days):
            print(
                f"  - {v.name}=={v.version} ({v.source or '<unknown source>'}) | "
                f"uploaded: {v.newest_upload.isoformat()} | age: {v.age_days:.2f} days",
                file=sys.stderr,
            )
        return 1

    checked = len([p for p in packages if isinstance(p, dict)])
    print(
        f"OK: checked {checked} packages in {lock_path} "
        f"with minimum age {args.min_age_days:.2f} day(s)."
    )
    if skipped_missing_times_non_pypi:
        print(
            f"WARNING: skipped {len(skipped_missing_times_non_pypi)} non-PyPI package(s) "
            "without upload-time metadata."
        )
    if missing_times_pypi:
        print(
            f"WARNING: skipped {len(missing_times_pypi)} PyPI package(s) without upload-time metadata "
            "(allowed by flag)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

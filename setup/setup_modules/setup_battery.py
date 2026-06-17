"""Battery setup helpers for lerobot-vulcan."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

StatusFn = Callable[[str], None]


@dataclass(frozen=True)
class BatterySetupOptions:
    """Configuration for the battery setup flow."""

    flash_golden: bool = True
    flash_profile: str = "df"
    verify_after_setup: bool = True


class BatterySetupManager:
    """Apply and verify Sourccey battery gauge configuration."""

    VALID_FLASH_PROFILES = {"df", "bq"}

    def __init__(
        self,
        project_root: Path,
        python_path: Path,
        print_status: StatusFn,
        print_success: StatusFn,
        print_warning: StatusFn,
        print_error: StatusFn,
    ) -> None:
        self.project_root = project_root
        self.python_path = python_path
        self.print_status = print_status
        self.print_success = print_success
        self.print_warning = print_warning
        self.print_error = print_error

        battery_dir = (
            self.project_root
            / "src"
            / "lerobot"
            / "scripts"
            / "sourccey"
            / "battery"
        )
        self.configure_script = battery_dir / "configure_bq34z100.py"
        self.check_script = battery_dir / "check_bq34z100.py"
        self.telemetry_script = battery_dir / "battery.py"
        self.golden_dir = battery_dir / "golden"

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _resolve_options(self, options: BatterySetupOptions) -> BatterySetupOptions:
        flash_profile = os.getenv("SOURCCEY_BQ34Z100_FLASH_PROFILE", options.flash_profile).strip().lower()
        if flash_profile not in self.VALID_FLASH_PROFILES:
            self.print_warning(
                f"Unsupported bq34z100 flash profile {flash_profile!r}; falling back to df."
            )
            flash_profile = "df"

        flash_golden = options.flash_golden
        if self._env_flag("SOURCCEY_SKIP_BQ34Z100_FLASH"):
            flash_golden = False
        if self._env_flag("SOURCCEY_FORCE_BQ34Z100_FLASH"):
            flash_golden = True

        verify_after_setup = options.verify_after_setup
        if self._env_flag("SOURCCEY_SKIP_BQ34Z100_VERIFY"):
            verify_after_setup = False

        return BatterySetupOptions(
            flash_golden=flash_golden,
            flash_profile=flash_profile,
            verify_after_setup=verify_after_setup,
        )

    def _profile_file(self, profile: str) -> Path:
        return self.golden_dir / f"0100_2_01-bq34z100.{profile}.fs"

    def _run_python_script(self, script_path: Path, args: list[str], label: str) -> subprocess.CompletedProcess[str]:
        command = [str(self.python_path), str(script_path), *args]
        self.print_status(f"{label}...")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=self.project_root,
        )
        return result

    def _emit_result_output(self, result: subprocess.CompletedProcess[str], *, success_label: str) -> bool:
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if result.returncode != 0:
            if stderr:
                self.print_error(stderr)
            elif stdout:
                self.print_error(stdout)
            return False

        if stdout:
            self.print_status(stdout)
        if stderr:
            self.print_warning(stderr)
        self.print_success(success_label)
        return True

    def _verify_setup(self) -> bool:
        verification_steps = [
            (
                self.configure_script,
                ["info"],
                "Verifying bq34z100 identity",
                "bq34z100 identity verified",
            ),
            (
                self.check_script,
                ["--pretty"],
                "Collecting bq34z100 diagnostics",
                "bq34z100 diagnostics collected",
            ),
            (
                self.telemetry_script,
                [],
                "Reading bq34z100 runtime telemetry",
                "bq34z100 runtime telemetry verified",
            ),
        ]

        for script_path, args, label, success_label in verification_steps:
            result = self._run_python_script(script_path, args, label)
            if not self._emit_result_output(result, success_label=success_label):
                return False

        return True

    def setup_bq34z100(self, options: BatterySetupOptions | None = None) -> bool:
        """Apply Sourccey battery setup and optional golden-image flash."""
        if options is None:
            options = BatterySetupOptions()
        options = self._resolve_options(options)

        if not self.configure_script.exists():
            self.print_warning(
                f"bq34z100 setup skipped: script not found at {self.configure_script}"
            )
            return True
        if not self.check_script.exists():
            self.print_warning(
                f"bq34z100 verification skipped: script not found at {self.check_script}"
            )
            return True
        if not self.telemetry_script.exists():
            self.print_warning(
                f"bq34z100 telemetry check skipped: script not found at {self.telemetry_script}"
            )
            return True
        if not self.python_path.exists():
            self.print_error(
                f"bq34z100 setup failed: venv Python not found at {self.python_path}"
            )
            return False
        if os.name != "nt" and not Path("/dev/i2c-1").exists():
            self.print_warning("bq34z100 setup skipped: /dev/i2c-1 not found")
            return True

        info_result = self._run_python_script(
            self.configure_script,
            ["info"],
            "Confirming bq34z100 communication",
        )
        if not self._emit_result_output(
            info_result,
            success_label="bq34z100 communication confirmed",
        ):
            return False

        used_flash = False
        if options.flash_golden:
            profile_file = self._profile_file(options.flash_profile)
            if profile_file.exists():
                flash_result = self._run_python_script(
                    self.configure_script,
                    ["flash-golden", "--profile", options.flash_profile],
                    f"Flashing bq34z100 golden profile ({options.flash_profile})",
                )
                if not self._emit_result_output(
                    flash_result,
                    success_label=f"bq34z100 golden profile ({options.flash_profile}) flashed",
                ):
                    return False
                used_flash = True
            else:
                self.print_warning(
                    f"Golden profile file not found at {profile_file}. "
                    "Falling back to starter battery configuration."
                )

        if not used_flash:
            configure_result = self._run_python_script(
                self.configure_script,
                ["setup-4s-lifepo4"],
                "Applying bq34z100 starter configuration",
            )
            if not self._emit_result_output(
                configure_result,
                success_label="bq34z100 starter configuration applied",
            ):
                return False

        if options.verify_after_setup and not self._verify_setup():
            return False

        if used_flash:
            self.print_success("bq34z100 setup completed with golden profile flash.")
        else:
            self.print_success("bq34z100 setup completed with starter configuration.")
        return True


def setup_bq34z100(
    project_root: Path,
    python_path: Path,
    print_status: StatusFn,
    print_success: StatusFn,
    print_warning: StatusFn,
    print_error: StatusFn,
    options: BatterySetupOptions | None = None,
) -> bool:
    """Convenience function for running the battery setup flow."""
    manager = BatterySetupManager(
        project_root=project_root,
        python_path=python_path,
        print_status=print_status,
        print_success=print_success,
        print_warning=print_warning,
        print_error=print_error,
    )
    return manager.setup_bq34z100(options=options)

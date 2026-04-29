# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rollout log throttling helpers."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

_DEFAULT_SPAM_PATTERNS = (
    "No new data available within timeout.",
    "Record loop is running slower (",
)


class RolloutSpamThrottleFilter(logging.Filter):
    """Throttle repetitive rollout warnings while preserving first occurrence."""

    def __init__(
        self,
        *,
        throttle_interval_s: float = 5.0,
        enabled: bool = True,
        patterns: tuple[str, ...] = _DEFAULT_SPAM_PATTERNS,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__()
        self.enabled = enabled
        self.throttle_interval_s = throttle_interval_s
        self.patterns = patterns
        self.clock = clock
        self._last_emit_by_pattern: dict[str, float] = {}
        self._suppressed_by_pattern: dict[str, int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.enabled:
            return True

        message = record.getMessage()
        matched_pattern = next((pattern for pattern in self.patterns if pattern in message), None)
        if matched_pattern is None:
            return True

        now = self.clock()
        last_emit = self._last_emit_by_pattern.get(matched_pattern)
        if last_emit is not None and (now - last_emit) < self.throttle_interval_s:
            self._suppressed_by_pattern[matched_pattern] = self._suppressed_by_pattern.get(matched_pattern, 0) + 1
            return False

        suppressed = self._suppressed_by_pattern.pop(matched_pattern, 0)
        if suppressed > 0:
            record.msg = f"{message} (suppressed {suppressed} similar messages)"
            record.args = ()

        self._last_emit_by_pattern[matched_pattern] = now
        return True


def configure_rollout_log_throttling(
    *,
    enabled: bool,
    throttle_interval_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Configure handler-level filters for rollout spam throttling.

    Handler-level attachment ensures logs from child/named loggers are also
    throttled when they propagate to root handlers.
    """
    root_logger = logging.getLogger()
    root_logger.filters = [f for f in root_logger.filters if not isinstance(f, RolloutSpamThrottleFilter)]
    for handler in root_logger.handlers:
        handler.filters = [f for f in handler.filters if not isinstance(f, RolloutSpamThrottleFilter)]

    if not enabled:
        return

    if not root_logger.handlers:
        # Fallback for unusual setups with no handlers.
        root_logger.addFilter(
            RolloutSpamThrottleFilter(throttle_interval_s=throttle_interval_s, clock=clock)
        )
        return

    for handler in root_logger.handlers:
        handler.addFilter(
            RolloutSpamThrottleFilter(throttle_interval_s=throttle_interval_s, clock=clock)
        )

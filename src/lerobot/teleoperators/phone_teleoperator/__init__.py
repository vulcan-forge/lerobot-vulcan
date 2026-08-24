try:  # prefer renamed remote config
    from .config_remote_teleoperator import PhoneTeleoperatorConfig
except ImportError:  # pragma: no cover - fallback for legacy name
    from .config_phone_teleoperator import PhoneTeleoperatorConfig  # type: ignore[import]
from .remote_teleoperator import PhoneTeleoperator
__all__ = ["PhoneTeleoperator", "PhoneTeleoperatorConfig"]

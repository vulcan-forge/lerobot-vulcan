from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config_remote_teleoperator import PhoneTeleoperatorConfig
    from .remote_teleoperator import PhoneTeleoperator

__all__ = ["PhoneTeleoperator", "PhoneTeleoperatorConfig"]


def __getattr__(name: str):
    if name == "PhoneTeleoperator":
        from .remote_teleoperator import PhoneTeleoperator

        return PhoneTeleoperator
    if name == "PhoneTeleoperatorConfig":
        try:  # prefer renamed remote config
            from .config_remote_teleoperator import PhoneTeleoperatorConfig
        except ImportError:  # pragma: no cover - fallback for legacy name
            from .config_phone_teleoperator import PhoneTeleoperatorConfig  # type: ignore[import]

        return PhoneTeleoperatorConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

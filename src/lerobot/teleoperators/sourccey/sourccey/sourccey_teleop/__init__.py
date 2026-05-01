from typing import TYPE_CHECKING

from .config_sourccey_teleop import PhoneTeleoperatorSourcceyConfig

if TYPE_CHECKING:
    from .sourccey_teleop import PhoneTeleoperatorSourccey

__all__ = ["PhoneTeleoperatorSourccey", "PhoneTeleoperatorSourcceyConfig"]


def __getattr__(name: str):
    if name == "PhoneTeleoperatorSourccey":
        from .sourccey_teleop import PhoneTeleoperatorSourccey

        return PhoneTeleoperatorSourccey
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

import json

from lerobot.robots.sourccey.sourccey.sourccey.sourccey_host import (
    DISCOVERY_MAGIC,
    DISCOVERY_ROBOT_TYPE,
    build_discovery_response_payload,
)


def test_sourccey_discovery_payload_matches_desktop_contract():
    payload = json.loads(build_discovery_response_payload().decode("utf-8"))

    assert payload == {
        "discovery_magic": DISCOVERY_MAGIC,
        "robot_type": DISCOVERY_ROBOT_TYPE,
    }

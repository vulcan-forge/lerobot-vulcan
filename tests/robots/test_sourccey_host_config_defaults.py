from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyHostConfig


def test_sourccey_host_config_keeps_default_network_ports():
    config = SourcceyHostConfig()

    assert config.port_zmq_cmd == 5555
    assert config.port_zmq_observations == 5556
    assert config.discovery_port == 42111
    assert config.watchdog_timeout_ms == 500

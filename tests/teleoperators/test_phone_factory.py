from unittest.mock import Mock, sentinel

from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config, phone


def test_upstream_phone_config_registers():
    assert TeleoperatorConfig.get_choice_class("phone") is phone.PhoneConfig


def test_phone_factory_uses_standard_device_discovery(monkeypatch):
    upstream_phone = Mock(return_value=sentinel.upstream_phone)
    monkeypatch.setattr(phone, "Phone", upstream_phone)
    phone_config = phone.PhoneConfig()

    assert make_teleoperator_from_config(phone_config) is sentinel.upstream_phone
    upstream_phone.assert_called_once_with(phone_config)

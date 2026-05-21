# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Helper to find the USB port associated with your MotorsBus.

Example:

```shell
lerobot-find-port
lerobot-find-port --disconnect-timeout-s 5
```
"""

import argparse
import platform
import time
from pathlib import Path


def find_available_ports():
    from lerobot.utils.import_utils import require_package

    require_package("pyserial", extra="hardware", import_name="serial")
    from serial.tools import list_ports

    if platform.system() == "Windows":
        # List COM ports using pyserial
        ports = [port.device for port in list_ports.comports()]
    else:  # Linux/macOS
        # List /dev/tty* ports for Unix-based systems
        ports = [str(path) for path in Path("/dev").glob("tty*")]
    return ports


def find_port(disconnect_timeout_s: float = 5.0, poll_interval_s: float = 0.1):
    print("Finding all available ports for the MotorsBus.")
    ports_before = find_available_ports()
    print("Ports before disconnecting:", ports_before)

    print("Remove the USB cable from your MotorsBus and press Enter when done.")
    input()  # Wait for user to disconnect the device

    deadline = time.monotonic() + disconnect_timeout_s
    ports_after = find_available_ports()
    ports_diff = list(set(ports_before) - set(ports_after))

    # Some OS/drivers take a moment to unregister the serial port after unplugging.
    while len(ports_diff) == 0 and time.monotonic() < deadline:
        time.sleep(poll_interval_s)
        ports_after = find_available_ports()
        ports_diff = list(set(ports_before) - set(ports_after))

    if len(ports_diff) == 1:
        port = ports_diff[0]
        print(f"The port of this MotorsBus is '{port}'")
        print("Reconnect the USB cable.")
    elif len(ports_diff) == 0:
        raise TimeoutError(
            "Could not detect the port before timeout. "
            f"No difference was found after waiting {disconnect_timeout_s:.1f}s."
        )
    else:
        raise OSError(f"Could not detect the port. More than one port was found ({ports_diff}).")


def main():
    parser = argparse.ArgumentParser(description="Find the USB port associated with your MotorsBus.")
    parser.add_argument(
        "--disconnect-timeout-s",
        type=float,
        default=5.0,
        help="How long to wait for the port to disappear after unplugging the board.",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=0.1,
        help="How often to poll for the disconnected port.",
    )
    args = parser.parse_args()
    find_port(
        disconnect_timeout_s=args.disconnect_timeout_s,
        poll_interval_s=args.poll_interval_s,
    )


if __name__ == "__main__":
    main()

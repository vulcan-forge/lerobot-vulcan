"""
Helper to list all available USB ports in a sorted, numbered format.
Returns data in a format suitable for Rust consumption.

Example:

```shell
lerobot-auto-config
```
"""

import platform
import json
from pathlib import Path


def find_available_ports():
    from serial.tools import list_ports  # Part of pyserial library

    if platform.system() == "Windows":
        # List COM ports using pyserial
        ports = [port.device for port in list_ports.comports()]
    else:  # Linux/macOS
        # List /dev/tty* ports for Unix-based systems
        ports = [str(path) for path in Path("/dev").glob("tty*")]
    return ports


def get_ports_list():
    """Return all available ports as a sorted list."""
    ports = find_available_ports()
    return sorted(ports)

def main():
    # For Rust consumption, output as JSON
    ports = get_ports_list()
    print(json.dumps(ports))


if __name__ == "__main__":
    main()

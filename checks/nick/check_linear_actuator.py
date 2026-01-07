import time
from lerobot.robots.sourccey.sourccey.sourccey_z_actuator.sourccey_z_actuator import SourcceyZActuator

z = SourcceyZActuator(raw_min=1800, raw_max=2000)
z.connect()

while True:
    r = z.read()
    pos = z.read_position_m100_100()
    print({"raw_10bit": r.raw_10bit, "raw": r.raw, "voltage": round(r.voltage, 4), "pos_m100_100": round(pos, 2)})
    time.sleep(0.2)

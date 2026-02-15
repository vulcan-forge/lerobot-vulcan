#!/usr/bin/env python3
# Read charge-related metrics from BQ34Z100 over I2C.
# Uses repeated-start reads to avoid byte-swap issues.

from __future__ import annotations

import argparse
from smbus2 import SMBus, i2c_msg

I2C_BUS_DEFAULT = 1
BQ_ADDR_DEFAULT = 0x55
CURRENT_SIGN_DEFAULT = -1  # flip sign so charging reads positive if desired

# Divider (board values)
R_TOP_OHMS = 249_000.0
R_BOTTOM_OHMS = 16_500.0
V_DIV_RATIO = R_BOTTOM_OHMS / (R_TOP_OHMS + R_BOTTOM_OHMS)

# Data Flash registers (for optional calibration dump)
REG_DF_CLASS = 0x3E
REG_DF_BLOCK = 0x3F
REG_BLOCKDATA_START = 0x40
REG_BLOCKDATA_CKSUM = 0x60
REG_BLOCKDATA_CTRL = 0x61

# Calibration Data subclass (TI datasheet)
CAL_SUBCLASS = 104

# Design Data subclass (contains Design Capacity)
DESIGN_SUBCLASS = 48

# Standard command maps seen on BQ34Z100 variants.
# "ti" matches the original map used in this repo (and gives sane V/T on your board).
# "sbs" is the SBS map used by the legacy tool.
CMD_MAPS: dict[str, dict[str, int]] = {
    "ti": {
        "soc": 0x02,
        "remaining": 0x04,
        "full": 0x06,
        "voltage": 0x08,
        "avg_current": 0x0A,
        "temperature": 0x0C,
        "flags": 0x0E,
        "current": 0x10,
    },
    "sbs": {
        "temperature": 0x08,
        "voltage": 0x09,
        "current": 0x0A,
        "avg_current": 0x0B,
        "soc": 0x0D,
        "remaining": 0x0F,
        "full": 0x10,
        "flags": 0x16,
    },
}


def _read_word(bus: SMBus, cmd: int) -> int:
    write = i2c_msg.write(BQ_ADDR, [cmd])
    read = i2c_msg.read(BQ_ADDR, 2)
    bus.i2c_rdwr(write, read)
    b = list(read)
    return b[0] | (b[1] << 8)


def _safe_read_word(bus: SMBus, cmd: int) -> tuple[int | None, str | None]:
    try:
        return _read_word(bus, cmd), None
    except OSError as e:
        return None, str(e)


def _s16(x: int) -> int:
    return x - 0x10000 if x & 0x8000 else x


def _df_read_block(bus: SMBus, subclass: int, block_index: int) -> bytes:
    bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_CTRL, 0x00)
    bus.write_byte_data(BQ_ADDR, REG_DF_CLASS, subclass)
    bus.write_byte_data(BQ_ADDR, REG_DF_BLOCK, block_index & 0xFF)
    return bytes(bus.read_i2c_block_data(BQ_ADDR, REG_BLOCKDATA_START, 32))


def _df_read_bytes(bus: SMBus, subclass: int, offset: int, length: int) -> bytes:
    block_index = offset // 32
    block = _df_read_block(bus, subclass, block_index)
    start = offset % 32
    return block[start:start + length]


def _ti_f4_decode(raw4: bytes) -> float:
    # TI "F4" format used in Calibration Data (see TI docs).
    r0, r1, r2, r3 = raw4
    exp = int(r0) - 128
    neg = (r1 & 0x80) != 0
    byte2 = r1 & 0x7F
    frac = float(byte2) + (float(r2) / 256.0) + (float(r3) / 65536.0)
    p = 8 - exp
    mag = (frac + 128.0) / (2.0 ** p)
    return -mag if neg else mag


def _soc_from_raw(soc_raw: int | None) -> tuple[float | None, str]:
    if soc_raw is None:
        return None, ""
    if soc_raw > 200:
        return soc_raw / 256.0, f" (raw {soc_raw}, Q8.8)"
    return float(soc_raw), ""


def _read_snapshot(
    bus: SMBus, cmds: dict[str, int], read_temp: bool
) -> tuple[dict[str, int | None], dict[str, str]]:
    snap: dict[str, int | None] = {}
    errors: dict[str, str] = {}

    for key in ("voltage", "current", "avg_current", "soc", "remaining", "full", "flags"):
        val, err = _safe_read_word(bus, cmds[key])
        snap[key] = val
        if err is not None:
            errors[key] = err

    if read_temp:
        val, err = _safe_read_word(bus, cmds["temperature"])
        snap["temperature"] = val
        if err is not None:
            errors["temperature"] = err
    else:
        snap["temperature"] = None

    return snap, errors


def _score_snapshot(snap: dict[str, int | None]) -> int:
    score = 0

    v = snap.get("voltage")
    if v is not None:
        if v < 2000:
            pack_v = (v / 1000.0) / V_DIV_RATIO
            if 6.0 <= pack_v <= 18.0:
                score += 3
            elif 6.0 <= pack_v <= 30.0:
                score += 1
            else:
                score -= 2
        else:
            if 6000 <= v <= 20000:
                score += 3
            elif 20000 < v <= 30000:
                score += 1
            else:
                score -= 2

    t = snap.get("temperature")
    if t is not None:
        c = (t / 10.0) - 273.15
        if -40.0 <= c <= 85.0:
            score += 3
        else:
            score -= 2

    soc_raw = snap.get("soc")
    soc_pct, _ = _soc_from_raw(soc_raw)
    if soc_pct is not None:
        if 0.0 <= soc_pct <= 120.0:
            score += 2
        else:
            score -= 1

    rem = snap.get("remaining")
    full = snap.get("full")
    if rem is not None and full is not None:
        if full == 0 and rem == 0:
            score -= 1
        elif 0 <= rem <= full * 1.2 and full <= 20000:
            score += 2
        else:
            score -= 1

    return score


def main() -> None:
    ap = argparse.ArgumentParser(description="Read charge metrics from BQ34Z100.")
    ap.add_argument("--bus", type=int, default=I2C_BUS_DEFAULT)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=BQ_ADDR_DEFAULT)
    ap.add_argument(
        "--map",
        choices=["auto"] + sorted(CMD_MAPS.keys()),
        default="auto",
        help="Command map to use (default: auto).",
    )
    ap.add_argument(
        "--current-sign",
        type=int,
        choices=[-1, 1],
        default=CURRENT_SIGN_DEFAULT,
        help="Multiply current by this sign (default: -1).",
    )
    ap.add_argument("--no-temp", action="store_true", help="Skip temperature read/output.")
    ap.add_argument("--cal", action="store_true", help="Dump calibration fields (subclass 104).")
    args = ap.parse_args()

    global BQ_ADDR
    BQ_ADDR = args.addr

    with SMBus(args.bus) as bus:
        if args.map == "auto":
            snapshots: dict[str, dict[str, int | None]] = {}
            scores: dict[str, int] = {}
            for name, cmds in CMD_MAPS.items():
                snap, _errs = _read_snapshot(bus, cmds, read_temp=not args.no_temp)
                snapshots[name] = snap
                scores[name] = _score_snapshot(snap)

            best_map = max(scores.items(), key=lambda kv: kv[1])[0]
            cmds = CMD_MAPS[best_map]
            snap = snapshots[best_map]
        else:
            cmds = CMD_MAPS[args.map]
            snap, _errs = _read_snapshot(bus, cmds, read_temp=not args.no_temp)
            best_map = args.map

        voltage_mv = snap["voltage"]
        temp_dK = snap["temperature"]
        curr_ma = _s16(snap["current"]) if snap["current"] is not None else None
        avg_ma = _s16(snap["avg_current"]) if snap["avg_current"] is not None else None
        if curr_ma is not None:
            curr_ma *= args.current_sign
        if avg_ma is not None:
            avg_ma *= args.current_sign
        soc_raw = snap["soc"]
        rem_raw = snap["remaining"]
        full_raw = snap["full"]
        flags = snap["flags"]

        design_cap_raw = _df_read_bytes(bus, DESIGN_SUBCLASS, 11, 2)
        design_cap = int.from_bytes(design_cap_raw, "big", signed=False)

        if args.cal:
            cc_gain_raw = _df_read_bytes(bus, CAL_SUBCLASS, 0, 4)
            cc_delta_raw = _df_read_bytes(bus, CAL_SUBCLASS, 4, 4)
            cc_offset_raw = _df_read_bytes(bus, CAL_SUBCLASS, 8, 2)
            board_offset_raw = _df_read_bytes(bus, CAL_SUBCLASS, 10, 1)
            vdiv_raw = _df_read_bytes(bus, CAL_SUBCLASS, 14, 2)

            cc_gain = _ti_f4_decode(cc_gain_raw)
            cc_delta = _ti_f4_decode(cc_delta_raw)
            cc_offset = int.from_bytes(cc_offset_raw, "big", signed=True)
            board_offset = int.from_bytes(board_offset_raw, "big", signed=True)
            vdiv = int.from_bytes(vdiv_raw, "little", signed=False)

    if args.map == "auto":
        print(f"Map: {best_map} (auto)")

    if voltage_mv is None:
        print("Voltage: <read failed>")
    # If the voltage looks like BAT (< 2 V), estimate pack using divider.
    elif voltage_mv < 2000:
        pack_v = (voltage_mv / 1000.0) / V_DIV_RATIO
        print(f"Voltage: {voltage_mv} mV (BAT) -> Pack ~ {pack_v:.2f} V")
    else:
        print(f"Voltage: {voltage_mv} mV (PACK)")
    if temp_dK is not None:
        print(f"Temperature: {temp_dK/10.0 - 273.15:.1f} C ({temp_dK} in 0.1K)")
    if curr_ma is None:
        print("Current: <read failed>")
    else:
        print(f"Current: {curr_ma} mA")
    if avg_ma is None:
        print("Avg Current: <read failed>")
    else:
        print(f"Avg Current: {avg_ma} mA")

    # SOC can show up as Q8.8 on some firmwares. If it looks too large, show Q8.8.
    soc_pct, soc_note = _soc_from_raw(soc_raw)
    if soc_pct is None:
        print("SOC: <read failed>")
    else:
        print(f"SOC: {soc_pct:.2f} %{soc_note}")

    # Capacity scaling heuristics for the TI map: some firmwares report in 10 mAh units.
    cap_scale = 1
    cap_note = ""
    if best_map == "ti" and full_raw is not None and design_cap >= 5000 and full_raw < 2000:
        cap_scale = 10
        cap_note = f" (raw {rem_raw}/{full_raw}, x{cap_scale})"
    rem_mah = rem_raw * cap_scale if rem_raw is not None else None
    full_mah = full_raw * cap_scale if full_raw is not None else None

    if cap_scale != 1:
        if rem_mah is None:
            print("Remaining Capacity: <read failed>")
        else:
            print(f"Remaining Capacity: {rem_mah} mAh{cap_note}")
        if full_mah is None:
            print("Full Charge Capacity: <read failed>")
        else:
            print(f"Full Charge Capacity: {full_mah} mAh{cap_note}")
    else:
        if rem_mah is None:
            print("Remaining Capacity: <read failed>")
        else:
            print(f"Remaining Capacity: {rem_mah} mAh")
        if full_mah is None:
            print("Full Charge Capacity: <read failed>")
        else:
            print(f"Full Charge Capacity: {full_mah} mAh")

    print(f"Design Capacity (DF): {design_cap} mAh")
    if flags is None:
        print("Flags: <read failed>")
    else:
        print(f"Flags: 0x{flags:04X}")

    if full_mah is not None and full_mah > 0 and rem_mah is not None:
        soc_from_cap = (rem_mah / float(full_mah)) * 100.0
        print(f"SOC (from capacity): {soc_from_cap:.2f} %")

    if args.cal:
        print("Calibration (DF subclass 104):")
        print(f"  CC Gain (F4): {cc_gain:.6g}")
        print(f"  CC Delta (F4): {cc_delta:.6g}")
        print(f"  CC Offset (I2): {cc_offset}")
        print(f"  Board Offset (I1): {board_offset}")
        print(f"  Voltage Divider (U2): {vdiv}")


if __name__ == "__main__":
    main()

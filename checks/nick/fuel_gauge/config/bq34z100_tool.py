#!/usr/bin/env python3
"""
bq34z100_tool.py - Dump + suggest + write BQ34Z100 data flash over SMBus/I2C (no BQStudio).

Key fix vs earlier version:
- Data Flash (BlockData 0x40..0x5F) 16-bit fields are interpreted as BIG-ENDIAN
  (matches your subclass dumps: e.g. 00 64 == 100, not 0x6400 == 25600).

Checksum rule unchanged:
- checksum = 255 - (sum(bytes) % 256)

This script assumes you are UNSEALED if you plan to write.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import struct
import math

try:
    from smbus2 import SMBus, i2c_msg
except ImportError:
    print("Missing dependency: smbus2. Install with: pip install smbus2", file=sys.stderr)
    sys.exit(2)


DEFAULT_ADDR = 0x55

# Command/register addresses (SMBus command codes)
REG_DF_CLASS = 0x3E
REG_DF_BLOCK = 0x3F
REG_BLOCKDATA_START = 0x40
REG_BLOCKDATA_CKSUM = 0x60
REG_BLOCKDATA_CTRL = 0x61


@dataclass(frozen=True)
class Field:
    name: str
    subclass: int
    offset: int
    ftype: str  # "U1", "I1", "U2", "I2", "S5", "S12", "H1", "H2"
    unit: str


FIELDS: List[Field] = [
    Field("Design Capacity", 48, 11, "I2", "mAh"),
    Field("Design Energy", 48, 13, "I2", "cWh"),
    Field("Cell Charge Voltage T1-T2", 48, 16, "U2", "mV"),
    Field("Cell Charge Voltage T2-T3", 48, 18, "U2", "mV"),
    Field("Volt Scale", 48, 60, "I1", "scale"),
    Field("Curr Scale", 48, 61, "I1", "scale"),
    Field("Energy Scale", 48, 62, "I1", "scale"),

    Field("Taper Current", 36, 0, "I2", "mA"),
    Field("Min Taper Capacity", 36, 2, "I2", "mAh/256"),
    Field("Cell Taper Voltage", 36, 4, "I2", "mV"),
    Field("Current Taper Window", 36, 6, "U1", "s"),

    Field("Cell Terminate Voltage", 80, 53, "I2", "mV"),

    Field("Quit Current", 81, 4, "I2", "mA"),
    Field("Dsg Relax Time", 81, 6, "U2", "s"),
    Field("Chg Relax Time", 81, 8, "U1", "s"),

    Field("Number of series cell", 64, 7, "U1", "cells"),
]

# --- Add near the top of the file (constants section) ---

SBS_CMDS = {
    "Temperature": 0x08,        # 0.1 K
    "Voltage": 0x09,            # mV
    "Current": 0x0A,            # mA (signed)
    "AverageCurrent": 0x0B,     # mA (signed)
    "StateOfCharge": 0x0D,      # %
    "RemainingCapacity": 0x0F,  # mAh
    "FullChargeCapacity": 0x10, # mAh
    "Flags": 0x16,              # bitfield
}

def _safe_pow2(exp: int) -> float:
    # double can handle about 2**1023; anything beyond is effectively inf
    if exp > 1023 or exp < -1074:   # -1074 covers denormals
        return float("nan")
    return 2.0 ** exp

def _safe_pow10(exp: int) -> float:
    # 10**308 is near float max
    if exp > 308 or exp < -324:
        return float("nan")
    return 10.0 ** exp

def _s8(x): return x-256 if x & 0x80 else x
def _s16_be(b):
    v = (b[0]<<8) | b[1]
    return v-65536 if v & 0x8000 else v
def _s16_le(b):
    v = b[0] | (b[1]<<8)
    return v-65536 if v & 0x8000 else v
def _s24_be(b):
    v = (b[0]<<16) | (b[1]<<8) | b[2]
    return v-0x1000000 if v & 0x800000 else v

def f32_le(b): return struct.unpack("<f", b)[0]
def f32_be(b): return struct.unpack(">f", b)[0]

def try_f4_candidates(raw4: bytes) -> dict:
    b0,b1,b2,b3 = raw4
    out = {}

    # IEEE-754
    out["ieee_f32_be"] = f32_be(raw4)
    out["ieee_f32_le"] = f32_le(raw4)
    out["ieee_f32_wordswap"] = f32_le(bytes([b2,b3,b0,b1]))

    # exp8 + mant24 variants
    exp = _s8(b0); mant = _s24_be(bytes([b1,b2,b3]))
    out["exp8_mant24_be"] = float(mant) * _safe_pow2(exp)

    exp = _s8(b3); mant = _s24_be(bytes([b0,b1,b2]))
    out["mant24_be_exp8_last"] = float(mant) * _safe_pow2(exp)

    # s16 + s16 base-2
    e_be = _s16_be(bytes([b0,b1])); m_be = _s16_be(bytes([b2,b3]))
    out["s16e_be_s16m_be_pow2"] = float(m_be) * _safe_pow2(e_be)

    e_le = _s16_le(bytes([b0,b1])); m_le = _s16_le(bytes([b2,b3]))
    out["s16e_le_s16m_le_pow2"] = float(m_le) * _safe_pow2(e_le)

    # s16 + s16 base-10
    out["s16e_be_s16m_be_pow10"] = float(m_be) * _safe_pow10(e_be)
    out["s16e_le_s16m_le_pow10"] = float(m_le) * _safe_pow10(e_le)

    return out

def ti_f4_decode(raw4: bytes) -> float:
    """
    Decode bq34z100 'F4' using TI Host Calibration method format (SLUA640B).

    Encoding per TI (floating2Byte):
      raw[0] = exp + 128
      raw[1] = byte2 (MSB is sign)
      raw[2] = byte1
      raw[3] = byte0

    Reconstruction:
      exp = raw0 - 128
      sign = raw1 bit7
      frac = (byte2_no_sign) + byte1/256 + byte0/65536
      mod_val = (frac + 128) / 2^(8 - exp)
      value = +/- mod_val
    """
    if len(raw4) != 4:
        raise ValueError("F4 requires exactly 4 bytes")

    r0, r1, r2, r3 = raw4
    exp = int(r0) - 128

    neg = (r1 & 0x80) != 0
    byte2 = r1 & 0x7F  # clear sign bit for magnitude

    frac = float(byte2) + (float(r2) / 256.0) + (float(r3) / 65536.0)

    # denominator = 2^(8 - exp)
    p = 8 - exp
    # guard extreme exponents (shouldn't normally happen)
    if p > 1023:
        mag = 0.0
    elif p < -1074:
        mag = float("inf")
    else:
        mag = (frac + 128.0) / (2.0 ** p)

    return -mag if neg else mag


def ti_f4_encode(value: float) -> bytes:
    """
    Encode bq34z100 'F4' using TI Host Calibration method format (SLUA640B).
    """
    if value == 0.0:
        return bytes([0x80, 0x00, 0x00, 0x00])  # exp=0, mant=0 (safe)

    val = float(value)
    mod_val = -val if val < 0 else val

    exp = 0
    tmp = mod_val

    # TI adds (1 + 2^-25) before normalization
    tmp = tmp * (1.0 + (2.0 ** -25))

    if tmp < 0.5:
        while tmp < 0.5:
            tmp *= 2.0
            exp -= 1
    else:
        # note: TI code uses "else if (tmpVal <= 1.0) { while (tmpVal >= 1.0) ... }"
        # The inner loop condition is what matters: while tmp >= 1.0, divide and exp++
        while tmp >= 1.0:
            tmp /= 2.0
            exp += 1

    if exp > 127:
        exp = 127
    elif exp < -128:
        exp = -128

    tmp = (2.0 ** (8 - exp)) * mod_val - 128.0
    byte2 = int(math.floor(tmp))
    tmp = (2.0 ** 8) * (tmp - float(byte2))
    byte1 = int(math.floor(tmp))
    tmp = (2.0 ** 8) * (tmp - float(byte1))
    byte0 = int(math.floor(tmp))

    if val < 0:
        byte2 = byte2 | 0x80

    raw0 = (exp + 128) & 0xFF
    raw1 = byte2 & 0xFF
    raw2 = byte1 & 0xFF
    raw3 = byte0 & 0xFF
    return bytes([raw0, raw1, raw2, raw3])

def _u16_be(b: bytes) -> int:
    return (b[0] << 8) | b[1]


def _i16_be(b: bytes) -> int:
    v = _u16_be(b)
    return v - 0x10000 if v & 0x8000 else v


def _checksum_32(block: bytes) -> int:
    # TI: checksum = 255 - (sum(bytes 0x40..0x5F) mod 256)
    return (255 - (sum(block) & 0xFF)) & 0xFF


class BQ34Z100:
    def __init__(self, bus: int, addr: int = DEFAULT_ADDR):
        self.addr = addr
        self.bus = SMBus(bus)

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass

    def read_bytes(self, reg: int, n: int) -> bytes:
        write = i2c_msg.write(self.addr, [reg])
        read = i2c_msg.read(self.addr, n)
        self.bus.i2c_rdwr(write, read)
        return bytes(read)

    def write_bytes(self, reg: int, data: bytes):
        msg = i2c_msg.write(self.addr, bytes([reg]) + data)
        self.bus.i2c_rdwr(msg)

    # ---- Data flash block access ----
    def df_read_block(self, subclass: int, block_index: int) -> Tuple[bytes, int]:
        self.write_bytes(REG_BLOCKDATA_CTRL, bytes([0x00]))
        self.write_bytes(REG_DF_CLASS, bytes([subclass & 0xFF]))
        self.write_bytes(REG_DF_BLOCK, bytes([block_index & 0xFF]))

        block = self.read_bytes(REG_BLOCKDATA_START, 32)
        cksum = self.read_bytes(REG_BLOCKDATA_CKSUM, 1)[0]
        return block, cksum

    def df_write_block(self, subclass: int, block_index: int, new_block: bytes, verify: bool = True):
        if len(new_block) != 32:
            raise ValueError("new_block must be exactly 32 bytes")

        self.write_bytes(REG_BLOCKDATA_CTRL, bytes([0x00]))
        self.write_bytes(REG_DF_CLASS, bytes([subclass & 0xFF]))
        self.write_bytes(REG_DF_BLOCK, bytes([block_index & 0xFF]))

        self.write_bytes(REG_BLOCKDATA_START, new_block)

        new_cksum = _checksum_32(new_block)
        self.write_bytes(REG_BLOCKDATA_CKSUM, bytes([new_cksum]))

        if verify:
            blk2, c2 = self.df_read_block(subclass, block_index)
            if blk2 != new_block:
                raise RuntimeError(f"Verify failed: block content mismatch (subclass={subclass}, block={block_index})")
            if c2 != new_cksum:
                raise RuntimeError(f"Verify failed: checksum mismatch (read {c2:#02x}, expected {new_cksum:#02x})")

    def df_read_bytes(self, subclass: int, offset: int, length: int) -> bytes:
        out = bytearray()
        end = offset + length
        i = offset
        while i < end:
            block_index = i // 32
            block, _ = self.df_read_block(subclass, block_index)
            block_off = i % 32
            take = min(32 - block_off, end - i)
            out.extend(block[block_off:block_off + take])
            i += take
        return bytes(out)

    def df_write_bytes(self, subclass: int, offset: int, data: bytes, verify: bool = True):
        i = 0
        while i < len(data):
            abs_off = offset + i
            block_index = abs_off // 32
            block, _ = self.df_read_block(subclass, block_index)
            block = bytearray(block)

            block_off = abs_off % 32
            take = min(32 - block_off, len(data) - i)
            block[block_off:block_off + take] = data[i:i + take]
            self.df_write_block(subclass, block_index, bytes(block), verify=verify)
            i += take

    # --- Add inside your BQ34Z100 class ---

    def sbs_read_u16(self, cmd: int) -> int:
        """
        SMBus Read Word. Returns unsigned 16-bit.
        Handles byte swap for Raspberry Pi SMBus behavior.
        """
        w = self.bus.read_word_data(self.addr, cmd)
        return ((w & 0xFF) << 8) | ((w >> 8) & 0xFF)

    def sbs_read_i16(self, cmd: int) -> int:
        """
        SMBus Read Word. Returns signed 16-bit.
        """
        v = self.sbs_read_u16(cmd)
        return v - 0x10000 if v & 0x8000 else v

    def sbs_snapshot(self) -> dict:
        out = {}

        out["Voltage_mV"] = self.sbs_read_u16(SBS_CMDS["Voltage"])
        out["RemainingCapacity_mAh"] = self.sbs_read_u16(SBS_CMDS["RemainingCapacity"])
        out["FullChargeCapacity_mAh"] = self.sbs_read_u16(SBS_CMDS["FullChargeCapacity"])
        out["StateOfCharge_pct"] = self.sbs_read_u16(SBS_CMDS["StateOfCharge"])
        out["Flags"] = f"0x{self.sbs_read_u16(SBS_CMDS['Flags']):04x}"

        out["Current_mA"] = self.sbs_read_i16(SBS_CMDS["Current"])
        out["AverageCurrent_mA"] = self.sbs_read_i16(SBS_CMDS["AverageCurrent"])

        t_dK = self.sbs_read_u16(SBS_CMDS["Temperature"])
        out["Temperature_C"] = (t_dK / 10.0) - 273.15

        return out


def decode_field(g: BQ34Z100, f: Field) -> Any:
    if f.ftype in ("U1", "I1"):
        b = g.df_read_bytes(f.subclass, f.offset, 1)
        v = b[0]
        if f.ftype == "I1" and v & 0x80:
            v = v - 0x100
        return v

    if f.ftype in ("U2", "I2", "H2"):
        b = g.df_read_bytes(f.subclass, f.offset, 2)
        if f.ftype in ("U2", "H2"):
            return _u16_be(b)
        return _i16_be(b)

    if f.ftype == "H1":
        b = g.df_read_bytes(f.subclass, f.offset, 1)
        return b[0]

    if f.ftype.startswith("S"):
        ln = int(f.ftype[1:])
        b = g.df_read_bytes(f.subclass, f.offset, ln)
        return b.split(b"\x00", 1)[0].decode("ascii", errors="replace")

    return None


def suggest_for_pack(series_cells: int, capacity_ah: float, nominal_v: float) -> Dict[str, Any]:
    capacity_mah = int(round(capacity_ah * 1000.0))
    # This tool keeps "Design Energy" as cWh (centi-Wh) for consistency with earlier output.
    # 12.8V * 10Ah = 128Wh => 12800 cWh
    design_energy_cwh = int(round((nominal_v * capacity_ah) * 100.0))

    # Safe-ish starting points; confirm with your cell + charger + BMS limits:
    cell_charge_mv = 3650
    cell_term_mv = 2900
    taper_ma = max(100, int(round(capacity_mah / 20)))  # C/20 => 500mA for 10Ah
    quit_ma = 100

    return {
        "Design Capacity": {"recommended": capacity_mah, "unit": "mAh"},
        "Design Energy": {"recommended": design_energy_cwh, "unit": "cWh"},
        "Number of series cell": {"recommended": series_cells, "unit": "cells"},
        "Cell Charge Voltage T1-T2": {"recommended": cell_charge_mv, "unit": "mV"},
        "Cell Charge Voltage T2-T3": {"recommended": cell_charge_mv, "unit": "mV"},
        "Cell Terminate Voltage": {"recommended": cell_term_mv, "unit": "mV"},
        "Taper Current": {"recommended": taper_ma, "unit": "mA"},
        "Quit Current": {"recommended": quit_ma, "unit": "mA"},
    }


def main():
    ap = argparse.ArgumentParser(description="BQ34Z100 data flash dump/suggest/write tool (no BQStudio).")
    ap.add_argument("--bus", type=int, default=1, help="I2C bus number (default 1 => /dev/i2c-1)")
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=DEFAULT_ADDR, help="7-bit I2C addr (default 0x55)")
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    ap.add_argument("--dump-subclass", type=lambda x: int(x, 0), action="append",
                    help="Dump full 32-byte blocks of a subclass (e.g. --dump-subclass 48). Can repeat.")
    ap.add_argument("--blocks", type=int, default=3, help="How many 32-byte blocks to dump per subclass (default 3)")
    ap.add_argument("--read-fields", action="store_true", help="Read and decode common important fields")
    ap.add_argument("--suggest", action="store_true", help="Suggest values for a given pack")
    ap.add_argument("--series", type=int, default=4, help="Series cells for --suggest (default 4)")
    ap.add_argument("--capacity-ah", type=float, default=10.0, help="Pack capacity in Ah for --suggest (default 10.0)")
    ap.add_argument("--nominal-v", type=float, default=12.8, help="Pack nominal voltage for --suggest (default 12.8)")
    ap.add_argument("--write", action="append",
                    help=("Write a value: name=value OR subclass:offset:type=value. "
                          "Examples: --write 'Design Capacity=10000' OR --write '48:11:I2=10000'"))
    ap.add_argument("--dry-run", action="store_true", help="Show what would be written, but do not write")
    ap.add_argument("--no-verify", action="store_true", help="Skip readback verification after writes")
    ap.add_argument("--peek", action="append", help="Peek raw bytes: subclass:offset:length (e.g. --peek 48:11:2)")
    ap.add_argument("--peek-f4", action="append", help="Peek a TI F4 float: subclass:offset (reads 4 bytes). Example: --peek-f4 104:0")
    ap.add_argument(
        "--peek-f4-candidates",
        dest="peek_f4_candidates",
        action="append",
        help="Peek 4 bytes and print candidate decodes: subclass:offset (e.g. 104:0)"
    )
    ap.add_argument(
        "--read-sbs",
        action="store_true",
        help="Read SBS (Smart Battery System) standard registers"
    )

    args = ap.parse_args()

    g = BQ34Z100(args.bus, args.addr)
    try:
        out: Dict[str, Any] = {"i2c": {"bus": args.bus, "addr": hex(args.addr)}}

        if args.dump_subclass:
            dumps = {}
            for sc in args.dump_subclass:
                blocks = []
                for bi in range(args.blocks):
                    blk, cks = g.df_read_block(sc, bi)
                    blocks.append({"block": bi, "checksum": f"0x{cks:02x}", "data_hex": blk.hex()})
                dumps[str(sc)] = blocks
            out["subclass_dumps"] = dumps

        values: Dict[str, Any] = {}
        if args.read_fields:
            for f in FIELDS:
                try:
                    values[f.name] = decode_field(g, f)
                except Exception as e:
                    values[f.name] = f"<error: {e}>"
            out["fields"] = values

        if args.suggest:
            out["suggested"] = suggest_for_pack(args.series, args.capacity_ah, args.nominal_v)

        if args.peek:
            peeks = []
            for p in args.peek:
                sc_s, off_s, ln_s = [x.strip() for x in p.split(":")]
                sc = int(sc_s, 0)
                off = int(off_s, 0)
                ln = int(ln_s, 0)
                b = g.df_read_bytes(sc, off, ln)
                peeks.append({"subclass": sc, "offset": off, "length": ln, "hex": b.hex()})
            out["peek"] = peeks

        if args.peek_f4:
            pf = []
            for item in args.peek_f4:
                sc_s, off_s = [x.strip() for x in item.split(":")]
                sc = int(sc_s, 0)
                off = int(off_s, 0)
                b = g.df_read_bytes(sc, off, 4)
                pf.append({
                    "subclass": sc,
                    "offset": off,
                    "raw_hex": b.hex(),
                    "value": ti_f4_decode(b),
                })
            out["peek_f4"] = pf

        if args.peek_f4_candidates:
            cands = []
            for item in args.peek_f4_candidates:
                sc_s, off_s = [x.strip() for x in item.split(":")]
                sc = int(sc_s, 0); off = int(off_s, 0)
                raw = g.df_read_bytes(sc, off, 4)
                cands.append({
                    "subclass": sc,
                    "offset": off,
                    "raw_hex": raw.hex(),
                    "candidates": try_f4_candidates(raw),
                })
            out["peek_f4_candidates"] = cands

        if args.read_sbs:
            snap = g.sbs_snapshot()
            if args.json:
                out["sbs"] = snap
            else:
                print("\nSBS snapshot:")
                for k, v in snap.items():
                    print(f"  - {k}: {v}")

        # Apply writes
        if args.write:
            by_name = {f.name: f for f in FIELDS}
            planned: List[Dict[str, Any]] = []

            for w in args.write:
                w = w.strip()
                if "=" not in w:
                    raise ValueError(f"Bad --write '{w}', expected ...=...")
                lhs, rhs = w.split("=", 1)
                lhs = lhs.strip()
                rhs = rhs.strip()

                if ":" in lhs:
                    sc_s, off_s, ty = [p.strip() for p in lhs.split(":")]
                    sc = int(sc_s, 0)
                    off = int(off_s, 0)
                    ftype = ty
                    f = Field(name=f"{sc}:{off}:{ftype}", subclass=sc, offset=off, ftype=ftype, unit="")
                else:
                    if lhs not in by_name:
                        raise ValueError(f"Unknown field name '{lhs}'. Known: {', '.join(by_name.keys())}")
                    f = by_name[lhs]

                # parse rhs based on type
                if f.ftype in ("U1", "I1", "H1"):
                    v = int(rhs, 0)
                    b = bytes([(v & 0xFF)])

                elif f.ftype in ("U2", "I2", "H2"):
                    v = int(rhs, 0)
                    # BIG-ENDIAN for data flash
                    b = bytes([((v >> 8) & 0xFF), (v & 0xFF)])

                elif f.ftype.startswith("S"):
                    ln = int(f.ftype[1:])
                    s = rhs.encode("ascii", errors="replace")[:ln]
                    b = s + b"\x00" * (ln - len(s))

                elif f.ftype == "F4":
                    v = float(rhs)
                    b = ti_f4_encode(v)

                else:
                    raise ValueError(f"Unsupported type for writing: {f.ftype}")

                old_b = g.df_read_bytes(f.subclass, f.offset, len(b))
                planned.append({
                    "field": f.name,
                    "subclass": f.subclass,
                    "offset": f.offset,
                    "type": f.ftype,
                    "old_hex": old_b.hex(),
                    "new_hex": b.hex(),
                    "new_value": rhs,
                })

                if not args.dry_run:
                    g.df_write_bytes(f.subclass, f.offset, b, verify=not args.no_verify)

            out["writes"] = planned
            out["write_mode"] = "dry-run" if args.dry_run else "applied"

        if args.json:
            print(json.dumps(out, indent=2, sort_keys=True))
        else:
            print(f"I2C bus={args.bus} addr={hex(args.addr)}")

            if "fields" in out:
                print("\nDecoded fields:")
                for k, v in out["fields"].items():
                    print(f"  - {k}: {v}")

            if "suggested" in out:
                print("\nSuggested updates for your pack:")
                for k, info in out["suggested"].items():
                    print(f"  - {k}: {info['recommended']} {info['unit']}")

            if "peek" in out:
                print("\nPeek:")
                for p in out["peek"]:
                    print(f"  - subclass {p['subclass']} offset {p['offset']} len {p['length']}: {p['hex']}")

            if "peek_f4" in out:
                print("\nPeek F4:")
                for p in out["peek_f4"]:
                    print(f"  - subclass {p['subclass']} offset {p['offset']}: raw={p['raw_hex']} value={p['value']}")

            if "writes" in out:
                print("\nWrites:")
                for w in out["writes"]:
                    print(f"  - {w['field']} @ subclass {w['subclass']} offset {w['offset']} ({w['type']}): "
                          f"{w['old_hex']} -> {w['new_hex']} ({w['new_value']})")
                print(f"Write mode: {out['write_mode']}")

            if "subclass_dumps" in out:
                print("\nSubclass dumps:")
                for sc, blocks in out["subclass_dumps"].items():
                    print(f"  Subclass {sc}:")
                    for b in blocks:
                        print(f"    block {b['block']} cks={b['checksum']} data={b['data_hex']}")

    finally:
        g.close()


if __name__ == "__main__":
    main()

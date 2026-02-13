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

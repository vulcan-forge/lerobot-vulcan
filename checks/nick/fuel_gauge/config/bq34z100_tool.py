#!/usr/bin/env python3
"""
bq34z100_tool.py - Dump + suggest + write BQ34Z100 data flash over SMBus/I2C.

Tested approach follows TI TRM data-flash block access:
- BlockDataControl (0x61) = 0x00
- DataFlashClass (0x3E) = subclass
- DataFlashBlock (0x3F) = block index (offset//32)
- BlockData (0x40..0x5F) 32 bytes
- BlockDataChecksum (0x60) checksum = 255 - (sum(block_bytes) % 256)

Refs:
- BQ34Z100-R2 TRM (data flash interface + checksum rule)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

try:
    from smbus2 import SMBus, i2c_msg
except ImportError:
    print("Missing dependency: smbus2. Install with: pip install smbus2", file=sys.stderr)
    sys.exit(2)


# 7-bit I2C address is usually 0x55 for TI gauges.
DEFAULT_ADDR = 0x55

# Command/register addresses (SMBus command codes)
REG_CNTL = 0x00  # Control() (requires 2-byte subcommand)
REG_DF_CLASS = 0x3E
REG_DF_BLOCK = 0x3F
REG_BLOCKDATA_START = 0x40
REG_BLOCKDATA_END = 0x5F
REG_BLOCKDATA_CKSUM = 0x60
REG_BLOCKDATA_CTRL = 0x61

# Common standard command registers (little-endian 16-bit reads)
REG_TEMPERATURE = 0x02
REG_VOLTAGE = 0x04
REG_FLAGS = 0x06
REG_AVGCURRENT = 0x10
REG_REMCAP = 0x0F
REG_FULLCHGCAP = 0x10  # NOTE: on some gauges AvgCurrent is 0x10; on bq34z100 AvgCurrent is 0x10/0x11.
# We'll avoid assuming too much here; we mainly use data flash.

# --- Data flash fields we care about (from TRM Data Flash Summary) ---
# Subclass IDs and offsets below are from bq34z100-R2 TRM Table 7-1 excerpts:
# - Config Data subclass 48: Design Capacity offset 11, Design Energy offset 13,
#   Cell Charge Voltage offsets 17/19 (T1-T2, T2-T3), and scale factors offsets 60/61/62, etc.
# - Charge Termination subclass 36: Taper Current offset 0, Min Taper Capacity offset 2,
#   Cell Taper Voltage offset 4, Current Taper Window offset 6.
# - IT Cfg subclass 80: Cell Terminate Voltage offset 53.
# - Current Thresholds subclass 81: Quit Current offset 4.

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
    Field("Cell Charge Voltage T1-T2", 48, 17, "U2", "mV"),
    Field("Cell Charge Voltage T2-T3", 48, 19, "U2", "mV"),
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


def _u16_le(b: bytes) -> int:
    return b[0] | (b[1] << 8)

def _i16_le(b: bytes) -> int:
    v = _u16_le(b)
    return v - 0x10000 if v & 0x8000 else v

def _checksum_32(block: bytes) -> int:
    # TI TRM: checksum is (255 - x) where x is 8-bit sum of bytes 0x40..0x5F
    return (255 - (sum(block) & 0xFF)) & 0xFF


class BQ34Z100:
    def __init__(self, bus: int, addr: int = DEFAULT_ADDR):
        self.bus_id = bus
        self.addr = addr
        self.bus = SMBus(bus)

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass

    def read_bytes(self, reg: int, n: int) -> bytes:
        # SMBus block read (command + repeated start) behavior:
        # smbus2 read_i2c_block_data uses SMBus "block read" with length prefix on some adapters.
        # We'll do raw I2C messages for exactness.
        write = i2c_msg.write(self.addr, [reg])
        read = i2c_msg.read(self.addr, n)
        self.bus.i2c_rdwr(write, read)
        return bytes(read)

    def write_bytes(self, reg: int, data: bytes):
        msg = i2c_msg.write(self.addr, bytes([reg]) + data)
        self.bus.i2c_rdwr(msg)

    def read_u2(self, reg: int, signed: bool = False) -> int:
        b = self.read_bytes(reg, 2)
        return _i16_le(b) if signed else _u16_le(b)

    # ---- Data flash block access ----
    def df_read_block(self, subclass: int, block_index: int) -> Tuple[bytes, int]:
        # Enable data flash access mode
        self.write_bytes(REG_BLOCKDATA_CTRL, bytes([0x00]))
        # Select subclass
        self.write_bytes(REG_DF_CLASS, bytes([subclass & 0xFF]))
        # Select block
        self.write_bytes(REG_DF_BLOCK, bytes([block_index & 0xFF]))
        # Read 32 bytes
        block = self.read_bytes(REG_BLOCKDATA_START, 32)
        cksum = self.read_bytes(REG_BLOCKDATA_CKSUM, 1)[0]
        return block, cksum

    def df_write_block(self, subclass: int, block_index: int, new_block: bytes, verify: bool = True):
        if len(new_block) != 32:
            raise ValueError("new_block must be exactly 32 bytes")
        # Must already be unsealed in practice; we won't try to brute-force keys.
        self.write_bytes(REG_BLOCKDATA_CTRL, bytes([0x00]))
        self.write_bytes(REG_DF_CLASS, bytes([subclass & 0xFF]))
        self.write_bytes(REG_DF_BLOCK, bytes([block_index & 0xFF]))

        # Write new bytes 0x40..0x5F
        self.write_bytes(REG_BLOCKDATA_START, new_block)

        # Write checksum
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
        start = offset
        end = offset + length
        i = start
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
    if f.ftype in ("U2", "I2", "H1", "H2"):
        n = 2
        b = g.df_read_bytes(f.subclass, f.offset, n)
        if f.ftype == "U2" or f.ftype == "H2":
            return _u16_le(b)
        if f.ftype == "H1":
            return b[0]  # if a header byte stored as 1, but we treat as raw
        return _i16_le(b)
    if f.ftype.startswith("S"):
        # ASCII string with fixed length after "S"
        ln = int(f.ftype[1:])
        b = g.df_read_bytes(f.subclass, f.offset, ln)
        return b.split(b"\x00", 1)[0].decode("ascii", errors="replace")
    return None


def suggest_for_pack(values: Dict[str, Any], series_cells: int, capacity_ah: float, nominal_v: float) -> Dict[str, Any]:
    capacity_mah = int(round(capacity_ah * 1000.0))
    design_energy_cwh = int(round((nominal_v * capacity_ah) * 100.0))  # Wh * 100 = cWh

    # Conservative generic starting points (you should confirm per your charger/cell datasheet):
    # - LFP typical full-charge per cell: 3.55–3.65V. We'll default to 3.65V.
    # - Terminate voltage per cell: often 2.8–3.0V depending on BMS cutoff philosophy.
    # - Taper current: recommend C/20 for "full" detection if charger allows (10Ah -> 500mA).
    # - Quit current: should be above your system standby draw so RELAX can happen; 100mA is common-ish.
    cell_charge_mv = 3650
    cell_term_mv = 2900
    taper_ma = max(100, int(round(capacity_mah / 20)))  # C/20
    quit_ma = 100

    return {
        "Design Capacity": {"recommended": capacity_mah, "unit": "mAh"},
        "Design Energy": {"recommended": design_energy_cwh, "unit": "cWh"},
        "Number of series cell": {"recommended": series_cells, "unit": "cells"},
        "Cell Charge Voltage T1-T2": {"recommended": cell_charge_mv, "unit": "mV", "note": "Confirm vs your LFP cell datasheet/charger CV setpoint."},
        "Cell Charge Voltage T2-T3": {"recommended": cell_charge_mv, "unit": "mV", "note": "Confirm vs your LFP cell datasheet/charger CV setpoint."},
        "Cell Terminate Voltage": {"recommended": cell_term_mv, "unit": "mV", "note": "Confirm vs your pack/BMS low-voltage cutoff target."},
        "Taper Current": {"recommended": taper_ma, "unit": "mA", "note": "Full detection depends on current staying below this for long enough; match your charger termination behavior."},
        "Quit Current": {"recommended": quit_ma, "unit": "mA", "note": "Set slightly ABOVE your system standby draw so the gauge can enter RELAX."},
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
            rec = suggest_for_pack(values, args.series, args.capacity_ah, args.nominal_v)
            out["suggested"] = rec

        # Apply writes
        if args.write:
            # Build lookup by friendly name
            by_name = {f.name: f for f in FIELDS}
            planned: List[Dict[str, Any]] = []

            for w in args.write:
                w = w.strip()
                # Either "Name=value" or "subclass:offset:type=value"
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
                    # minimal field stub
                    f = Field(name=f"{sc}:{off}:{ftype}", subclass=sc, offset=off, ftype=ftype, unit="")
                else:
                    if lhs not in by_name:
                        raise ValueError(f"Unknown field name '{lhs}'. Use one of: {', '.join(by_name.keys())}")
                    f = by_name[lhs]

                # parse rhs based on type
                if f.ftype in ("U1", "I1"):
                    v = int(rhs, 0)
                    b = bytes([(v & 0xFF)])
                elif f.ftype in ("U2", "I2", "H2"):
                    v = int(rhs, 0)
                    b = bytes([(v & 0xFF), ((v >> 8) & 0xFF)])
                elif f.ftype.startswith("S"):
                    ln = int(f.ftype[1:])
                    s = rhs.encode("ascii", errors="replace")[:ln]
                    b = s + b"\x00" * (ln - len(s))
                else:
                    raise ValueError(f"Unsupported type for writing: {f.ftype}")

                # Read old
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
            # Pretty print
            print(f"I2C bus={args.bus} addr={hex(args.addr)}")
            if "fields" in out:
                print("\nDecoded fields:")
                for k, v in out["fields"].items():
                    print(f"  - {k}: {v}")
            if "suggested" in out:
                print("\nSuggested updates for your pack:")
                for k, info in out["suggested"].items():
                    note = f"  ({info.get('note')})" if "note" in info else ""
                    print(f"  - {k}: {info['recommended']} {info['unit']}{note}")
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

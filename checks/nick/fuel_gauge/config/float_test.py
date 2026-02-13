import sys
from bq34z100_tool import BQ34Z100  # adjust import if needed
import struct
import math

BUS = 1
ADDR = 0x55

def s8(x): return x-256 if x & 0x80 else x
def s16_be(b):
    v = (b[0]<<8) | b[1]
    return v-65536 if v & 0x8000 else v

def ieee_be(b): return struct.unpack(">f", b)[0]
def ieee_le(b): return struct.unpack("<f", b)[0]

def ti_f4_decode_candidate(raw4):
    r0,r1,r2,r3 = raw4
    exp = r0 - 128
    if exp > 127: exp -= 256
    neg = (r1 & 0x80) != 0
    byte2 = r1 & 0x7F
    frac = byte2 + r2/256.0 + r3/65536.0
    mag = (frac + 128.0) / (2.0 ** (8 - exp))
    return -mag if neg else mag

def dump_f4(g, offset):
    raw = g.df_read_bytes(104, offset, 4)
    print(f"\nOffset {offset}")
    print("Raw:", raw.hex())
    print("IEEE BE:", ieee_be(raw))
    print("IEEE LE:", ieee_le(raw))
    print("TI F4 candidate:", ti_f4_decode_candidate(raw))

def main():
    g = BQ34Z100(BUS, ADDR)
    dump_f4(g, 0)
    dump_f4(g, 4)

if __name__ == "__main__":
    main()

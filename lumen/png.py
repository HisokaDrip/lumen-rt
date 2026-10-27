"""Minimal PNG encoder. No Pillow, just zlib + struct."""
import struct
import zlib

import numpy as np


def _chunk(tag, data):
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def encode_png(pixels):
    """pixels: (h, w, 3) uint8 array -> PNG bytes."""
    px = np.ascontiguousarray(pixels, dtype=np.uint8)
    if px.ndim != 3 or px.shape[2] != 3:
        raise ValueError(f"expected (h, w, 3) array, got {px.shape}")
    h, w, _ = px.shape

    # every scanline gets a leading filter byte; 0 = none. zlib does the real work anyway
    raw = np.zeros((h, w * 3 + 1), dtype=np.uint8)
    raw[:, 1:] = px.reshape(h, w * 3)

    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw.tobytes(), 9))
        + _chunk(b"IEND", b"")
    )


def write_png(path, pixels):
    with open(path, "wb") as f:
        f.write(encode_png(pixels))

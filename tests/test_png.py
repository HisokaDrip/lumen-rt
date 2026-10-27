import struct
import zlib

import numpy as np
import pytest

from lumen.png import encode_png


def read_chunks(data):
    pos = 8
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        (crc,) = struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF
        yield tag, body
        pos += 12 + n


def test_roundtrip_pixels():
    rng = np.random.default_rng(1)
    px = rng.integers(0, 256, size=(7, 5, 3), dtype=np.uint8)
    data = encode_png(px)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"

    chunks = dict(read_chunks(data))
    w, h = struct.unpack(">II", chunks[b"IHDR"][:8])
    assert (w, h) == (5, 7)

    raw = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), dtype=np.uint8).reshape(7, 16)
    assert (raw[:, 0] == 0).all()
    np.testing.assert_array_equal(raw[:, 1:].reshape(7, 5, 3), px)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4), dtype=np.uint8))

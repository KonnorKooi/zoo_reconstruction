"""
tag_dict.py

Shared AprilTag dictionary for the zoo reconstruction pipeline.

Builds a custom OpenCV ArUco dictionary from the bit patterns used by
generate_apriltags.py (IDs 0-14, custom spiral encoding) and
generate_more_apriltags.py (IDs 15-34, official OpenCV bits).

Imported by estimate_camera_pose.py and visualize_apriltags.py so the
codes and spiral logic live in exactly one place.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Bit codes — one per tag ID, in spiral order
# IDs 0-14 source: generate_apriltags.py (custom spiral from tag36h11.c codes)
# IDs 15-34 source: generate_more_apriltags.py (via cv2.aruco.DICT_APRILTAG_36h11)
# ---------------------------------------------------------------------------
TAG36H11_CODES: list[int] = [
    0x0000000d7e00984b,  # ID  0
    0x0000000dda664ca7,  # ID  1
    0x0000000dc4a1c821,  # ID  2
    0x0000000e17b470e9,  # ID  3
    0x0000000ef91d01b1,  # ID  4
    0x0000000f429cdd73,  # ID  5
    0x000000005da29225,  # ID  6
    0x00000001106cba43,  # ID  7
    0x0000000223bed79d,  # ID  8
    0x000000021f51213c,  # ID  9
    0x000000033eb19ca6,  # ID 10
    0x00000003f76eb0f8,  # ID 11
    0x0000000469a97414,  # ID 12
    0x000000045dcfe0b0,  # ID 13
    0x00000004a6465f72,  # ID 14
    0x000000076540cc8a,  # ID 15
    0x00000001d16e5f2b,  # ID 16
    0x0000000339a9187b,  # ID 17
    0x0000000895a227f8,  # ID 18
    0x0000000565fdb845,  # ID 19
    0x0000000f4a0e9386,  # ID 20
    0x00000002f2132b07,  # ID 21
    0x00000007c239dd95,  # ID 22
    0x0000000e8675a26d,  # ID 23
    0x00000003be8089ae,  # ID 24
    0x0000000d62b3c7ad,  # ID 25
    0x00000000fac6fece,  # ID 26
    0x00000005e2e4699c,  # ID 27
    0x0000000b9f4eeca2,  # ID 28
    0x0000000477574423,  # ID 29
    0x00000006db930173,  # ID 30
    0x0000000598bde796,  # ID 31
    0x00000001c509378c,  # ID 32
    0x000000089551687c,  # ID 33
    0x0000000a71972d9c,  # ID 34
]

_DATA_SIDE = 6
_NBITS = _DATA_SIDE * _DATA_SIDE


def _clockwise_spiral(n: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    top, bottom, left, right = 0, n - 1, 0, n - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            positions.append((c, top))
        top += 1
        for r in range(top, bottom + 1):
            positions.append((right, r))
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                positions.append((c, bottom))
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                positions.append((left, r))
            left += 1
    return positions


_SPIRAL = _clockwise_spiral(_DATA_SIDE)


def code_to_bits(code: int) -> np.ndarray:
    """Return a 6×6 uint8 array from a spiral-encoded 36-bit code."""
    grid = np.zeros((_DATA_SIDE, _DATA_SIDE), dtype=np.uint8)
    for bit_idx, (dc, dr) in enumerate(_SPIRAL):
        grid[dr][dc] = (code >> (_NBITS - 1 - bit_idx)) & 1
    return grid


def build_custom_dictionary() -> cv2.aruco.Dictionary:
    """Build an OpenCV ArUco dictionary matching all 35 printed tags (IDs 0-34)."""
    all_bytes = [
        cv2.aruco.Dictionary.getByteListFromBits(code_to_bits(code))[0]
        for code in TAG36H11_CODES
    ]
    return cv2.aruco.Dictionary(np.stack(all_bytes, axis=0), _DATA_SIDE, 0)

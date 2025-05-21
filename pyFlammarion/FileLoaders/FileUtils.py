import struct
from typing import  BinaryIO

def write_null_terminated_string(file: BinaryIO, string: str) -> None:
    """Write a null-terminated string to a file."""
    file.write(string.encode("ascii") + b"\0")


def write_uint32(file: BinaryIO, value: int) -> None:
    """Write a 32-bit unsigned integer to a file."""
    file.write(struct.pack("<I", value))


def write_char(file: BinaryIO, value: str) -> None:
    """Write a single character to a file."""
    file.write(value.encode("ascii"))


def write_double(file: BinaryIO, value: float) -> None:
    """Write a double-precision float to a file."""
    file.write(struct.pack("<d", value))
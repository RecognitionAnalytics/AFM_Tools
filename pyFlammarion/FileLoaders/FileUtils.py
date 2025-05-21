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
    

def read_null_terminated_string(open_file: BinaryIO) -> str:
    """
    Read a null-terminated string from a binary file.
    
    Parameters
    ----------
    open_file : BinaryIO
        An open file object.
        
    Returns
    -------
    str
        The string read from the file.
    """
    result = bytearray()
    while True:
        char = open_file.read(1)
        if char == b'\0' or not char:  # End of string or end of file
            break
        result.extend(char)
    return result.decode('utf-8')

def read_char(open_file: BinaryIO) -> str:
    """
    Read a single character from a binary file.
    
    Parameters
    ----------
    open_file : BinaryIO
        An open file object.
        
    Returns
    -------
    str
        The character read from the file.
    """
    return open_file.read(1).decode('utf-8')

def read_uint32(open_file: BinaryIO) -> int:
    """
    Read an unsigned 32-bit integer from a binary file.
    
    Parameters
    ----------
    open_file : BinaryIO
        An open file object.
        
    Returns
    -------
    int
        The integer read from the file.
    """
    return struct.unpack('<I', open_file.read(4))[0]

def read_double(open_file: BinaryIO) -> float:
    """
    Read a double-precision float from a binary file.
    
    Parameters
    ----------
    open_file : BinaryIO
        An open file object.
        
    Returns
    -------
    float
        The float value read from the file.
    """
    return struct.unpack('<d', open_file.read(8))[0]
    
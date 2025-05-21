#rewrite of code from https://github.com/AFM-SPM/AFMReader
#code is under the GNU General Public License v3.0
#due to copy and changing

from .FlammarionFile  import FlammarionFile , FlammarionImageData
from pathlib import Path
from typing import Any, BinaryIO, Dict,  Union
from .FileUtils import write_null_terminated_string, write_uint32, write_char, write_double
from .FileUtils import read_null_terminated_string, read_uint32, read_char, read_double
import numpy as np
import re
import struct 

def save_to_gwy(image: 'FlammarionFile', file_path: Union[Path, str], verbose=False) -> None:
    """
    Save a FlammarionFile to a Gwyddion (.gwy) file
    
    Only supports saving the first image in the FlammarionFile. I may fix this in the future.
    
    Parameters:
    -----------
    image_data : FlammarionFile
        The image data object to be saved
    file_path : Union[Path, str]
        Path where the GWY file should be saved
    """
    if image is None or image.images is None:
        raise ValueError("No image data to save")
    
    # Convert to Path object
    file_path = Path(file_path)
    
    # Ensure file has .gwy extension
    if not str(file_path).lower().endswith('.gwy'):
        file_path = file_path.with_suffix('.gwy')
    
    
    yres, xres = image.resolution
    
    # Get physical dimensions in meters
    xreal, yreal = image.physicalSize  
    
    
    with open(file_path, "wb") as f:
        # Write header
        f.write(b"GWYP")
        cc=0
        for image_data in image.images:
            # Create the data structure
            if verbose:
                print(f"Writing image {image_data.label} to GWY file")
            layerMeta = {}
            for key, value in image.metaData.items():
                if key not in layerMeta:
                    layerMeta[key] = value
            for key, value in image_data.metaData.items():
                layerMeta[key] = value
            
            data_dict = {
                f"/{cc}/data": {
                    "object_type": "GwyDataField",
                    "xres": xres,
                    "yres": yres,
                    "xreal": xreal,
                    "yreal": yreal,
                    "si_unit_xy": {"object_type": "GwySIUnit", "unitstr": image.physicalSizeUnit},
                    "si_unit_z": {"object_type": "GwySIUnit", "unitstr": image.physicalSizeUnit},
                    "data": image_data.data,
                },
                f"/{cc}/data/log": {},
                f"/{cc}/data/title": image_data.label or "AFM Data",
                f"/{cc}/meta": image.metaData,
            }
            
            # Add processing history if available
            #if image_data.processingHistory:
            #    data_dict["/0/data/log"]["processing"] = ", ".join(image_data.processingHistory)

            # Write main container
            _write_gwy_object(f, data_dict)
            cc += 1
    if verbose:
        print(f"GWY file saved to {file_path}")


def _write_gwy_component(file: BinaryIO, component_name: str, data: Any) -> None:
    """
    Write a GWY component to a file.

    Parameters:
    -----------
    file : BinaryIO
        Open file to write to
    component_name : str
        Name of the component
    data : Any
        Data to write
    """
    write_null_terminated_string(file, component_name)

    if isinstance(data, dict):
        # Write object
        write_char(file, "o")
        _write_gwy_object(file, data)
    elif isinstance(data, bool):
        # Write boolean
        write_char(file, "b")
        write_char(file, "1" if data else "0")
    elif isinstance(data, str):
        # Write string
        write_char(file, "s")
        write_null_terminated_string(file, data)
    elif isinstance(data, int):
        # Write 32-bit integer
        write_char(file, "i")
        write_uint32(file, data)
    elif isinstance(data, float):
        # Write double
        write_char(file, "d")
        write_double(file, data)
    elif isinstance(data, np.ndarray):
        # Write array of doubles (2D image data)
        write_char(file, "D")
        flattened_data = data.flatten()
        write_uint32(file, len(flattened_data))
        for value in flattened_data:
            write_double(file, float(value))


def _write_gwy_object(file: BinaryIO, data_dict: Dict[str, Any]) -> None:
    """
    Write a GWY object to a file.

    Parameters:
    -----------
    file : BinaryIO
        Open file to write to
    data_dict : Dict[str, Any]
        Dictionary of object data
    """
    if "object_type" in data_dict:
        object_type = data_dict.pop("object_type")
    else:
        object_type = "GwyContainer"  # Default object type

    write_null_terminated_string(file, object_type)

    # Calculate data size by writing to a temporary buffer first
    import io
 

def load_gwy(file_path: Path | str ) -> FlammarionFile:
    """
    Extract image and metadata from the .gwy file.

    Parameters
    ----------
    file_path : Path or str
        Path to the .gwy file.

    Returns
    -------
    FlammarionFile

    Raises
    ------
    FileNotFoundError
        If the file is not found.

    ```
    """
    
    file_path = Path(file_path)
    
    try:
        image_data_dict: dict[Any, Any] = {}
        with Path.open(file_path, "rb") as open_file:  # pylint: disable=unspecified-encoding
            # Read header
            _ = open_file.read(4)
            _gwy_read_object(open_file, data_dict=image_data_dict)
        
        newFile = FlammarionFile()
        newFile.filename = file_path
        newFile.datasetName = file_path.stem
        newFile.images = []
        # Extract channels by finding keys that match the pattern '/X/data'
        channel_ids = [int(re.match(r'/(\d+)/data', key).group(1)) for key in image_data_dict.keys() if re.match(r'/\d+/data', key)]
        channel_ids = list(set(channel_ids))  # Remove duplicates
        for channel_id in channel_ids:
            key_prefix = f"/{channel_id}"
            
            if f"{key_prefix}/data" in image_data_dict:
                data_field = image_data_dict[f"{key_prefix}/data"]
                
                
                # Create a new image layer
                image_layer = FlammarionImageData()
                image_layer.direction=''
                
                # Set image data
                if "data" in data_field:
                    image_layer.data = data_field["data"]
                
                # Set label if available
                if f"{key_prefix}/data/title" in image_data_dict:
                    image_layer.label = image_data_dict[f"{key_prefix}/data/title"]
                     
                elif f"{channel_id}/data/title" in image_data_dict: #handle read error in code 
                    image_layer.label = image_data_dict[f"{channel_id}/data/title"]
                else:
                    image_layer.label = f"Channel {channel_id}"
                
                
                    
                # Add processing history if available
                if f"{key_prefix}/data/log" in image_data_dict :
                    image_layer.processingHistory = image_data_dict[f"{key_prefix}/data/log"]
                    
                # Set physical dimensions
                if "xres" in data_field and "yres" in data_field:
                    newFile.resolution = (data_field["yres"], data_field["xres"])
                    image_layer.resolution = (data_field["yres"], data_field["xres"])
                
                if "xreal" in data_field and "yreal" in data_field:
                    newFile.physicalSize = (data_field["xreal"], data_field["yreal"])
                    image_layer.physicalSize = (data_field["yreal"], data_field["xreal"])
                    
                # Set units
                if "si_unit_xy" in data_field and "unitstr" in data_field["si_unit_xy"]:
                    newFile.physicalSizeUnit = data_field["si_unit_xy"]["unitstr"]
                
                if 'si_unit_z' in data_field and "unitstr" in data_field["si_unit_z"]:
                    image_layer.zUnit = data_field["si_unit_z"]["unitstr"]
                    
                # Set metadata
                image_layer.metaData = {}
                
                # Add layer-specific metadata if available
                if f"{key_prefix}/meta" in image_data_dict:
                    image_layer.metaData.update(image_data_dict[f"{key_prefix}/meta"])
                elif f'\x01/{channel_id}/meta' in image_data_dict:
                    image_layer.metaData.update(image_data_dict[f'\x01/{channel_id}/meta'])
                    
                if 'trace' in image_layer.metaData :
                    image_layer.direction = image_layer.metaData['trace']
                    
                # Add the image layer to the file
                newFile.images.append(image_layer)

        # Set global metadata
        newFile.metaData = {}
        for channel_id in channel_ids:
            if f"/{channel_id}/meta" in image_data_dict:
                newFile.metaData.update(image_data_dict[f"/{channel_id}/meta"])
            elif f'\x01/{channel_id}/meta' in image_data_dict:
                    image_layer.metaData.update(image_data_dict[f'\x01/{channel_id}/meta'])
    except FileNotFoundError:
        raise
    return newFile


def _gwy_read_object(open_file: BinaryIO, data_dict: dict) -> None:
    """
    Parse and extract data from a `.gwy` file object, starting at the current open file read position.

    Parameters
    ----------
    open_file : BinaryIO
        An open file object.
    data_dict : dict
        Dictionary of `.gwy` file image properties.
    """
    object_name = read_null_terminated_string(open_file=open_file)
    data_size = read_uint32(open_file)
    
    # Read components
    read_data_size = 0
    while read_data_size < data_size:
        component_data_size = _gwy_read_component(
            open_file=open_file,
            initial_byte_pos=open_file.tell(),
            data_dict=data_dict,
        )
        read_data_size += component_data_size


def _gwy_read_component(open_file: BinaryIO, initial_byte_pos: int, data_dict: dict) -> int:
    """
    Parse and extract data from a `.gwy` file object, starting at the current open file read position.

    Parameters
    ----------
    open_file : BinaryIO
        An open file object.
    initial_byte_pos : int
        Initial position, as byte.
    data_dict : dict
        Dictionary of `.gwy` file image properties.

    Returns
    -------
    int
        Size of the component in bytes.
    """
    component_name = read_null_terminated_string(open_file=open_file)
    data_type = _read_gwy_component_dtype(open_file=open_file)

    if data_type == "o":
        sub_dict: dict[Any, Any] = {}
        _gwy_read_object(open_file=open_file, data_dict=sub_dict)
        data_dict[component_name] = sub_dict
    elif data_type == "c":
        value = read_char(open_file=open_file)
        data_dict[component_name] = value
    elif data_type == "i":
        value = read_uint32(open_file=open_file)  # type: ignore
        data_dict[component_name] = value
    elif data_type == "d":
        value = read_double(open_file=open_file)  # type: ignore
        data_dict[component_name] = value
    elif data_type == "s":
        value = read_null_terminated_string(open_file=open_file)
        data_dict[component_name] = value
    elif data_type == "D":
        array_size = read_uint32(open_file=open_file)
        data = np.zeros(array_size)
        for index in range(array_size):
            data[index] = read_double(open_file=open_file)
        if "xres" in data_dict and "yres" in data_dict:
            data = data.reshape((data_dict["xres"], data_dict["yres"]))
        data_dict["data"] = data

    return open_file.tell() - initial_byte_pos

def _read_gwy_component_dtype(open_file: BinaryIO) -> str:
    """
    Read the data type of a `.gwy` file component.

    Possible data types are as follows:

    - 'b': boolean
    - 'c': character
    - 'i': 32-bit integer
    - 'q': 64-bit integer
    - 'd': double
    - 's': string
    - 'o': `.gwy` format object

    Capitalised versions of some of these data types represent arrays of values of that data type. Arrays are stored as
    an unsigned 32 bit integer, describing the size of the array, followed by the unseparated array values:

    - 'C': array of characters
    - 'I': array of 32-bit integers
    - 'Q': array of 64-bit integers
    - 'D': array of doubles
    - 'S': array of strings
    - 'O': array of objects.

    Parameters
    ----------
    open_file : BinaryIO
        An open file object.

    Returns
    -------
    str
        Python string (one character long) of the data type of the component's value.
    """
    return open_file.read(1).decode("ascii")


from .FlammarionFile  import FlammarionFile 
from pathlib import Path
from typing import Any, BinaryIO, Dict,  Union
from .FileUtils import write_null_terminated_string, write_uint32, write_char, write_double
import numpy as np



def save_to_gwy(image: 'FlammarionFile', file_path: Union[Path, str], verbose=False) -> None:
    """
    Save a FlammarionFile to a Gwyddion (.gwy) file
    
    Parameters:
    -----------
    image_data : FlammarionImaFlammarionFileeData
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
            write_gwy_object(f, data_dict)
            cc += 1
    if verbose:
        print(f"GWY file saved to {file_path}")


def write_gwy_component(file: BinaryIO, component_name: str, data: Any) -> None:
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
        write_gwy_object(file, data)
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


def write_gwy_object(file: BinaryIO, data_dict: Dict[str, Any]) -> None:
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

    temp_buffer = io.BytesIO()
    for key, value in data_dict.items():
        write_gwy_component(temp_buffer, key, value)

    # Write the data size
    data_size = temp_buffer.tell()
    write_uint32(file, data_size)

    # Write the actual data
    file.write(temp_buffer.getvalue())

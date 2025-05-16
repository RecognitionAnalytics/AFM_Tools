import os
import sys


# __init__.py for pyFlammarion package
# Imports all procedures from AFM_Tools.py to make them available through the package

import importlib.util

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Path to AFM_Tools.py (assuming it's in the parent directory AFM_Tools)
afm_tools_path = os.path.join(parent_dir, "AFM_Tools.py")

# Check if the file exists
if not os.path.exists(afm_tools_path):
    raise ImportError(f"Cannot load AFM_Tools.py from {afm_tools_path}")

# Load the module
spec = importlib.util.spec_from_file_location("afm_tools_module", afm_tools_path)
afm_tools_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(afm_tools_module)

# Import all attributes from the module to make them available when importing pyFlammarion
module_attributes = dir(afm_tools_module)
for attr in module_attributes:
    # Skip private attributes (starting with underscore)
    if not attr.startswith("_"):
        globals()[attr] = getattr(afm_tools_module, attr)

# Define what should be imported with "from pyFlammarion import *"
__all__ = [attr for attr in module_attributes if not attr.startswith("_")]

# Package metadata
__version__ = "0.1.0"
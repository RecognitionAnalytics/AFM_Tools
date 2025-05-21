 
from .FlamFlattening import *
from .FlamFiles import *
from .MathTools import *

import inspect
import types
from . import FlamFlattening as _imaging

# Automatically populate __all__ with all functions and classes from AFM_Tools
# __all__ = [
#     name for name, obj in inspect.getmembers(_imaging)
#     if isinstance(obj, (types.FunctionType, type))
# ]
# Package metadata
__version__ = "0.1.0"
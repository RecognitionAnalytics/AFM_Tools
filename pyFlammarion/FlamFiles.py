import os
from glob import glob
from pathlib import Path
from FileLoaders.FlammarionFile import FlammarionFile
from FileLoaders.gwyFile import save_to_gwy
from FileLoaders.miFiles import loadMI
                            
fileLoaders = {
    ".mi"   : loadMI,
    ".ibw"  : None,
    ".jpk"  : None,
    ".gwy"  : None,
}                            

fileSavers = {
    ".mi"   : None,
    ".ibw"  : None,
    ".jpk"  : None,
    ".gwy"  : save_to_gwy,
}

def knownLoadFileTypes():
    """
    Get the known file types for loading.

    Returns
    -------
    list
        A list of known file types.
    """
    return list(fileLoaders.keys())

def knownSaveFileTypes():
    """
    Get the known file types for saving.

    Returns
    -------
    list
        A list of known file types.
    """
    return list(fileSavers.keys())  
    

 
def loadFile(filename, verbose=False):
    """
    Load an afm file and return a dictionary with the data and parameters
    Parameters
    ----------
    filename : str
        The name of the file to load
    Returns
    -------
    data : dict
        A dictionary with the data and parameters
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")
    if not os.path.isfile(filename):
        raise ValueError(f"{filename} is not a file")
    for ext, loader in fileLoaders.items():
        if filename.lower().endswith(ext.lower()):
            if loader is None:
                raise ValueError(f"File type {ext} not supported")
            data: FlammarionFile= loader(filename)
            if verbose:
                print(f"Loaded {filename} with {ext} loader")
                for   value  in data.images:
                    print(f"{value.label} : {value.direction}")
            return data
        
    raise ValueError("File type not supported")

    
def saveFile(filename, data: FlammarionFile, verbose=False):
    """
    Save a file with the given data
    Parameters
    ----------
    filename : str
        The name of the file to save
    data : FlammarionFile
        The image to save
    Returns
    -------
    None
    """
    
    for ext, saver in fileSavers.items():
        if filename.lower().endswith(ext.lower()):
            if saver is None:
                raise ValueError(f"File type {ext} not supported")
            saver(data, filename)
            if verbose:
                print(f"Saved {filename} with {ext} saver")
            return
        
    raise ValueError("File type not supported") 
    
def get_all_files(folder_path, extension=".mi", recursive=False):
    """
    Get all .mi files from a specified folder

    Parameters:
    -----------
    folder_path : str
        Path to folder containing .mi files
    recursive : bool, optional
        Whether to search recursively in subfolders (default: False)

    Returns:
    --------
    list
        List of paths to .mi files
    """
    if recursive:
        pattern = os.path.join(folder_path, "**", f"*{extension}")
        return glob(pattern, recursive=True)
    else:
        pattern = os.path.join(folder_path, f"*{extension}")
        return glob(pattern)




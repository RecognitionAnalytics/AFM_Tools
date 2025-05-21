
import numpy as np
from typing import Dict, Any

class FlammarionImageData:
    def __init__(self):
        """
        Initialize a 2D array data container.

        This class represents a 2D array of data typically from AFM measurements with associated metadata.

        Attributes:
            data (np.array): 2D array containing the height/signal data.
            label (str): Label or name for the data.
            zunit (str): Unit for the height/signal values (e.g., 'nm', 'V').
           
            direction (str): Scan direction information.
            processingHistory (list[str]): Record of processing steps applied to the data.
        """
        self.data:np.array = None
        self.label:str = None
        self.zunit:str = None
        self.direction:str = None
        self.metaData: Dict[str, Any] = {}
        self.processingHistory: list[str] = []
        self.physicalSize:tuple[float,float] = (0.0, 0.0)
        self.physicalSizeUnit:str = "m"

class FlammarionFile:
    def __init__(self):
        self.filename:str = None
        self.datasetName:str = None
        self.images:list[FlammarionImageData] = []
        self.metaData: Dict[str, Any] = None
        self.resolution:tuple[int,int] = (0, 0)
        self.physicalSize:tuple[float,float] = (0.0, 0.0)
        self.physicalSizeUnit:str = "m"

    
    def __getitem__(self, key):
        """
        
        Parameters
        ----------
        key : str
            The key of the image to retrieve (format: label_direction).
            
            
        Returns
        -------
        FlammarionImageData
            The requested image data.
            
        Raises
        ------
        ValueError
            If the image with the specified key is not found.
        """
        return self.getImage(key)
        
    def __setitem__(self, key, value):
        """
        Parameters
        ----------
        key : str
            The key for the image (format: label_direction).
        value : FlammarionImageData
            The image data to assign.
            
        Raises
        ------
        TypeError
            If the value is not a FlammarionImageData object.
        """
        if not isinstance(value, FlammarionImageData):
            raise TypeError("Value must be a FlammarionImageData object")
        
        # Parse key to set label and direction
        if "_" in key:
            label, direction = key.split("_", 1)
            value.label = label
            value.direction = direction
        else:
            value.label = key
            value.direction = ""
        
        # Check if image with this key already exists
        for i, img in enumerate(self.images):
            if img.label + "_" + img.direction == key:
                # Replace existing image
                self.images[i] = value
                return
        
        # Add new image
        self.images.append(value)

    def __iter__(self):
        """
        Make the FlammarionFile iterable over image keys.
        
        Yields
        ------
        str
            Each image key in the format "label_direction".
        """
        for img in self.images:
            yield img.label + "_" + img.direction
    
    def imageKeys(self):
        """
        Get the keys of the images in the file.

        Returns
        -------
        list
            A list of keys for the images in the file.
        """
        return [img.label + "_" + img.direction for img in self.images]
    
    def getImages(self, searchString:str = None)-> 'FlammarionFile':
        
        subset:FlammarionImageData = []
        for img in self.images:
            if searchString is None or searchString.lower() in img.label.lower() or searchString.lower() in img.direction.lower():
                subset.append(img)
                
        subsetFile = FlammarionFile()
        subsetFile.filename = self.filename
        subsetFile.datasetName = self.datasetName
        subsetFile.metaData = self.metaData.copy()
        subsetFile.resolution = self.resolution
        subsetFile.physicalSize = self.physicalSize
        subsetFile.physicalSizeUnit = self.physicalSizeUnit + ""
        subsetFile.images = subset
        return subsetFile
        
        
    def getImage(self, key)-> FlammarionImageData:
        """
        Get an image from the file by its key.

        Parameters
        ----------
        key : str
            The key of the image to get.

        Returns
        -------
        FlammarionImageData
            The image data.
        """
        for img in self.images:
            if img.label + "_" + img.direction == key:
                return img
        raise ValueError(f"Image {key} not found")
    
    def __str__(self):
        """
        Return a string representation of the FlammarionFile.
        
        Returns
        -------
        str
            A human-readable string representation of the file contents.
        """
        file_info = f"FlammarionFile: {self.filename}\n"
        file_info += f"Dataset: {self.datasetName}\n"
        file_info += f"Resolution: {self.resolution[0]}x{self.resolution[1]}\n"
        file_info += f"Physical Size: {self.physicalSize[0]}x{self.physicalSize[1]} {self.physicalSizeUnit}\n"
        
        file_info += f"\nImages ({len(self.images)}):\n"
        for i, img in enumerate(self.images):
            file_info += f"  {i+1}. {img.label}_{img.direction} ({img.zunit})\n"
        print()
        if self.metaData and len(self.metaData) > 0:
            file_info += "\nMetadata:\n"
            for key, value in self.metaData.items():
                # Truncate long values for display
                if isinstance(value, str) and len(value) > 50:
                    value = value[:47] + "..."
                file_info += f"  {key}: {value}\n"
        
        return file_info
   
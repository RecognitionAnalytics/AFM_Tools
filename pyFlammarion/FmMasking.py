from enum import Enum
import numpy as np
from scipy import ndimage
from scipy.stats import trim_mean
from skimage.filters import threshold_multiotsu
from skimage.feature import canny
from .FileLoaders.FlammarionFile import FlammarionFile, FlammarionImageData

class EdgeDetectionMethod(Enum):
    Sobel = 1
    Prewitt = 2
    Roberts = 3    
    Canny =4 
    
class AFMMaskMethods(Enum):
    HighPoints = 1  # Remove regions that are too high
    StreakMask = 13  # Detect scan-direction streaks caused by persistent tip damage or contamination
    TrimmedMean = 3  # Mask the points that are well out of the normal from the image
    ParticleDetection = 4  # Find particles in the image and mask them
    Unsmoothable = 5  # Smooth image and remove the points that do not smooth well
    EdgeArtifacts = 7  # Remove or downweight edges where tip lift-off or overshoot occurs
    SpikeArtifacts = 10  # Mask single-pixel or very small clusters of anomalous height (z-spikes)
    LowPoints = 11  # Analogous to HighPoints but for unusually deep wells (e.g., pits or holes)
    SharpEdges = 12  # Mask sharp edges or features that are not smooth



def _sharpEdgesMask(imagePack: FlammarionImageData, threshold_factor=1.0, filter_method=EdgeDetectionMethod.Sobel):
    """
    Generate a mask that excludes sharp edges where the slope is higher than a threshold.
    
    Args:
        image (np.ndarray): 2D array of topography data
        threshold_factor (float): Factor multiplied by std of gradient to determine threshold
        filter_method (str): Method for calculating gradients (EdgeDetectionMethod.Sobel, EdgeDetectionMethod.Prewitt, dgeDetectionMethod.Roberts)
        **kwargs: Additional parameters (unused)
    
    Returns:
        np.ndarray: Boolean mask where 1 indicates good points to use for flattening
    """
    image = imagePack.data
    # Apply gradient filter based on specified method
    if filter_method  == EdgeDetectionMethod.Prewitt:
        grad_x = ndimage.prewitt(image, axis=1)
        grad_y = ndimage.prewitt(image, axis=0)
    elif filter_method  == EdgeDetectionMethod.Roberts:
        grad_x = ndimage.roberts(image, axis=1)
        grad_y = ndimage.roberts(image, axis=0)
    else:  # Default to Sobel
        grad_x = ndimage.sobel(image, axis=1)
        grad_y = ndimage.sobel(image, axis=0)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Calculate threshold based on standard deviation of gradient
    gradient_std = np.std(gradient_magnitude)
    threshold = threshold_factor * gradient_std
    
    # Create mask where points with gradient magnitude below threshold are marked as good (1)
    mask = np.where(gradient_magnitude < threshold, 1, 0)
    
    return mask

def _highPointsMask(imagePack: FlammarionImageData, threshold_percentile=90):
    """
    Generate a mask that excludes high points above a percentile threshold.
    
    Args:
        image (np.ndarray): 2D array of topography data
        threshold_percentile (float): Percentile threshold above which points are masked
        **kwargs: Additional parameters (unused)
    
    Returns:
        np.ndarray: Boolean mask where 1 indicates good points to use for flattening
    """
    image = imagePack.data
    # Calculate the threshold height based on percentile
    threshold = np.percentile(image, threshold_percentile)
    
    # Create mask where points below threshold are marked as good (1)
    mask = np.where(image < threshold, 1, 0)
    
    return mask

def _lowPointsMask(imagePack: FlammarionImageData, threshold_percentile=10):
    """
    Generate a mask that excludes low points below a percentile threshold.
    
    Args:
        image (np.ndarray): 2D array of topography data
        threshold_percentile (float): Percentile threshold below which points are masked
        **kwargs: Additional parameters (unused)
    
    Returns:
        np.ndarray: Boolean mask where 1 indicates good points to use for flattening
    """
    image = imagePack.data
    # Calculate the threshold height based on percentile
    threshold = np.percentile(image, threshold_percentile)
    
    # Create mask where points above threshold are marked as good (1)
    mask = np.where(image > threshold, 1, 0)
    
    return mask

def _streakMask(imagePack: FlammarionImageData, scan_direction='horizontal', std_threshold=2.0, min_length=10):
    """
    Generate a mask that excludes scan-direction streaks caused by tip damage or contamination.
    
    Args:
        image (np.ndarray): 2D array of topography data
        scan_direction (str): Direction of scan ('horizontal' or 'vertical')
        std_threshold (float): Number of standard deviations for identifying outlier lines
        min_length (int): Minimum streak length to be considered for masking
        **kwargs: Additional parameters (unused)
    
    Returns:
        np.ndarray: Boolean mask where 1 indicates good points to use for flattening
    """
    image = imagePack.data
    # Create initial mask with all points marked as good
    rows, cols = image.shape
    mask = np.ones_like(image)
    
    # Transpose for vertical scan direction
    if scan_direction.lower() == 'vertical':
        image = image.T
        mask = mask.T
        rows, cols = cols, rows
    
    # Calculate mean and standard deviation for each row
    row_means = np.nanmean(image, axis=1)
    row_stds = np.nanstd(row_means)
    overall_mean = np.nanmean(row_means)
    
    # Find outlier rows that might have streaks
    outlier_rows = np.where(np.abs(row_means - overall_mean) > std_threshold * row_stds)[0]
    
    # For each outlier row, look for streaks of similar height
    for row in outlier_rows:
        # Calculate local derivatives along the row
        if row < rows - 1:
            derivatives = np.abs(image[row, :] - image[row+1, :])
            
            # Areas with low derivatives are more likely to be streak artifacts
            streak_candidates = np.where(derivatives < np.nanpercentile(derivatives, 25))[0]
            
            # Group consecutive positions
            if len(streak_candidates) > 0:
                # Find gaps in the sequence
                gaps = np.where(np.diff(streak_candidates) > 1)[0]
                
                # Split into groups of consecutive positions
                streak_groups = np.split(streak_candidates, gaps + 1)
                
                # Mark long streaks in the mask
                for streak in streak_groups:
                    if len(streak) >= min_length:
                        mask[row, streak] = 0
    
    # Transpose back if necessary
    if scan_direction.lower() == 'vertical':
        mask = mask.T
    
    return mask

def _trimmedMeanMask(imagePack: FlammarionImageData, trim_factor=3.0):
    """
    Generate a mask that excludes points that are far from the trimmed mean.
    
    Args:
        image (np.ndarray): 2D array of topography data
        trim_factor (float): Factor multiplied by MAD to determine outlier threshold
        **kwargs: Additional parameters (unused)
    
    Returns:
        np.ndarray: Boolean mask where 1 indicates good points to use for flattening
    """
    image = imagePack.data
    # Calculate trimmed mean (removing 10% from top and bottom)
    flat_img = image.flatten()
    trimmed_mean = trim_mean(flat_img, 0.1)
    
    # Calculate Median Absolute Deviation (MAD) - robust measure of dispersion
    median = np.median(flat_img)
    mad = np.median(np.abs(flat_img - median))
    
    # Define upper and lower thresholds based on MAD
    upper_threshold = trimmed_mean + trim_factor * mad
    lower_threshold = trimmed_mean - trim_factor * mad
    
    # Create mask
    mask = np.ones_like(image)
    mask[image > upper_threshold] = 0
    mask[image < lower_threshold] = 0
    
    return mask

def _particleDetectionMask(imagePack: FlammarionImageData, threshold_method='otsu', min_size=10, **kwargs):
    """
    Generate a mask that excludes particles (features that stand out from background).
    
    Args:
        image (np.ndarray): 2D array of topography data
        threshold_method (str): Method to determine threshold ('otsu' or 'percentile')
        min_size (int): Minimum size of particle to mask
        **kwargs: Additional parameters (e.g., percentile for 'percentile' method)
    
    Returns:
        np.ndarray: Boolean mask where 1 indicates good points to use for flattening
    """
    image = imagePack.data
    # Determine threshold
    if threshold_method.lower() == 'otsu':
        try:
            threshold = threshold_multiotsu(image, classes=2)[0]
        except:
            # Fallback if Otsu fails
            threshold = np.percentile(image, 75)
    else:  # percentile method
        percentile = kwargs.get('percentile', 75)
        threshold = np.percentile(image, percentile)
    
    # Create initial binary mask for particles
    binary = image > threshold
    
    # Label connected components
    labeled, num_features = ndimage.label(binary)
    
    # Filter small particles
    mask = np.ones_like(image)
    for i in range(1, num_features + 1):
        particle_size = np.sum(labeled == i)
        if particle_size >= min_size:
            mask[labeled == i] = 0
            
    return mask

def _unsmoothableMask(imagePack: FlammarionImageData, filter_size=5, threshold_factor=2.0):
    """
    Generate a mask that excludes areas that don't smooth well (sharp features).
    
    Args:
        image (np.ndarray): 2D array of topography data
        filter_size (int): Size of smoothing filter
        threshold_factor (float): Factor to determine threshold for masking
        **kwargs: Additional parameters (unused)
    
    Returns:
        np.ndarray: Boolean mask where 1 indicates good points to use for flattening
    """
    image = imagePack.data
    # Apply Gaussian filter for smoothing
    smoothed = ndimage.gaussian_filter(image, sigma=filter_size/6)
    
    # Calculate absolute difference between original and smoothed
    diff = np.abs(image - smoothed)
    
    # Calculate threshold based on statistics of the difference
    threshold = np.mean(diff) + threshold_factor * np.std(diff)
    
    # Create mask
    mask = np.ones_like(image)
    mask[diff > threshold] = 0
    
    return mask

def _edgeArtifactsMask(imagePack: FlammarionImageData, edge_width=5, method=EdgeDetectionMethod.Sobel, threshold=0.5, **kwargs):
    """
    Generate a mask that excludes the edges of the image and detected features edges.
    
    Args:
        image (np.ndarray): 2D array of topography data
        edge_width (int): Width of image border to mask
        method (str): Edge detection method ('sobel' or 'canny')
        threshold (float): Threshold for edge detection (0-1 for sobel, absolute value for canny)
        **kwargs: Additional parameters (sigma for gaussian smoothing, etc.)
    
    Returns:
        np.ndarray: Boolean mask where 1 indicates good points to use for flattening
    """
    image = imagePack.data
    rows, cols = image.shape
    mask = np.ones_like(image)
    
    # Mask image borders
    mask[:edge_width, :] = 0  # Top
    mask[-edge_width:, :] = 0  # Bottom
    mask[:, :edge_width] = 0  # Left
    mask[:, -edge_width:] = 0  # Right
    
    # Apply optional Gaussian smoothing before edge detection
    sigma = kwargs.get('sigma', 1.0)
    smoothed_img = ndimage.gaussian_filter(image, sigma=sigma)
    
    # Detect edges using specified method
    if method  == EdgeDetectionMethod.Canny:
        # Get min/max for automatic thresholding if needed
        vmin, vmax = np.percentile(image, [1, 99])
        low_threshold = kwargs.get('low_threshold', threshold * (vmax - vmin) + vmin)
        high_threshold = kwargs.get('high_threshold', low_threshold * 3)
        
        edges = canny(smoothed_img, 
                     low_threshold=low_threshold,
                     high_threshold=high_threshold)
    else:  # Default to Sobel
        # Calculate Sobel gradients
        sx = ndimage.sobel(smoothed_img, axis=0)
        sy = ndimage.sobel(smoothed_img, axis=1)
        
        # Calculate gradient magnitude
        sobel_magnitude = np.sqrt(sx**2 + sy**2)
        
        # Normalize to 0-1
        sobel_magnitude = sobel_magnitude / np.max(sobel_magnitude)
        
        # Threshold to get edges
        edges = sobel_magnitude > threshold
    
    # Dilate edges slightly to ensure they're fully masked
    dilate_size = kwargs.get('dilate_size', 2)
    if dilate_size > 0:
        edges = ndimage.binary_dilation(edges, iterations=dilate_size)
    
    # Combine edge detection with border mask
    mask[edges] = 0
    
    return mask

def _spikeArtifactsMask(imagePack: FlammarionImageData, threshold_sigma=5.0, neighborhood_size=3):
    """
    Generate a mask that excludes spike artifacts (isolated extreme values).
    
    Args:
        image (np.ndarray): 2D array of topography data
        threshold_sigma (float): Number of standard deviations for spike detection
        neighborhood_size (int): Size of neighborhood for local comparison
        **kwargs: Additional parameters (unused)
    
    Returns:
        np.ndarray: Boolean mask where 1 indicates good points to use for flattening
    """
    image = imagePack.data
    # Create initial mask with all points marked as good
    mask = np.ones_like(image)
    
    # Calculate global statistics
    global_mean = np.nanmean(image)
    global_std = np.nanstd(image)
    
    # First pass: mark extreme global outliers
    extreme_mask = np.abs(image - global_mean) > threshold_sigma * global_std
    
    # Second pass: compare to local neighborhood
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if extreme_mask[i, j]:
                # Define neighborhood boundaries
                i_min = max(0, i - neighborhood_size)
                i_max = min(image.shape[0], i + neighborhood_size + 1)
                j_min = max(0, j - neighborhood_size)
                j_max = min(image.shape[1], j + neighborhood_size + 1)
                
                # Extract neighborhood excluding the center point
                neighborhood = image[i_min:i_max, j_min:j_max].copy()
                neighborhood[i-i_min, j-j_min] = np.nan  # Exclude center point
                
                # Calculate local statistics
                local_mean = np.nanmean(neighborhood)
                local_std = np.nanstd(neighborhood)
                
                # Mark as spike if significantly different from neighborhood
                if np.abs(image[i, j] - local_mean) > threshold_sigma * local_std:
                    mask[i, j] = 0
    
    return mask
    
 
mask_functions = {
    AFMMaskMethods.HighPoints: _highPointsMask,
    AFMMaskMethods.StreakMask: _streakMask,
    AFMMaskMethods.TrimmedMean: _trimmedMeanMask,
    AFMMaskMethods.ParticleDetection: _particleDetectionMask,
    AFMMaskMethods.Unsmoothable: _unsmoothableMask,
    AFMMaskMethods.EdgeArtifacts: _edgeArtifactsMask,
    AFMMaskMethods.SpikeArtifacts: _spikeArtifactsMask,
    AFMMaskMethods.LowPoints: _lowPointsMask,
    AFMMaskMethods.SharpEdges: _sharpEdgesMask
}    

def MaskImage(imagePack: FlammarionImageData , maskMethod=AFMMaskMethods.Unsmoothable, **kwargs) ->np.array:    
    """
    Generate a mask for AFM image flattening using the specified method.
    Args:
        image (np.ndarray): 2D array of topography data
        method (AFMMaskGenerator): Method to use for generating the mask
        **kwargs: Additional parameters specific to the mask generation method
    Returns:
        np.ndarray: Boolean mask where 1 indicates good points to use for flattening
    """
    # Look up the appropriate function and call it with the image and kwargs
    if maskMethod in mask_functions:
        mask = mask_functions[maskMethod](imagePack, **kwargs)
    else:
        raise ValueError(f"Unsupported mask method: {maskMethod}")
    
    return mask    
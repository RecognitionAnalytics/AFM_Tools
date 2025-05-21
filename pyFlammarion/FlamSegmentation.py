import numpy as np
from scipy import ndimage
from MathTools import SetPrefix
import pandas as pd
from skimage import  segmentation,  measure
from skimage.filters import threshold_otsu
from skimage.filters import threshold_multiotsu
from enum import Enum
from skimage.feature import canny
class EdgeDetectionMethod(Enum):
    Sobel = 1
    Prewitt = 2
    Roberts = 3    
    Canny = 4 
    
#Function to calculate segment statistics
def analyze_segment(img, mask, img_dict):
    segment_data = img[mask]
    
    # Get unit and apply scaling
    unit = img_dict.get('unit', 'um')
    if unit == 'um':
        # Already in micrometers, no need to scale
        scale_factor = 1
        prefix = 'μ'
    else:
        # Default scaling
        scale_factor = 1
        prefix = ''
    
    # Calculate basic statistics
    avg_height = np.mean(segment_data) * scale_factor
    rms_roughness = np.std(segment_data) * scale_factor
    mean_roughness = np.mean(np.abs(segment_data - avg_height)) * scale_factor
    
    # Calculate skewness
    if len(segment_data) > 1:
        skew = np.mean(((segment_data - avg_height) / rms_roughness) ** 3) if rms_roughness > 0 else 0
        # Calculate kurtosis
        kurtosis = np.mean(((segment_data - avg_height) / rms_roughness) ** 4) - 3 if rms_roughness > 0 else 0
    else:
        skew = 0
        kurtosis = 0
    
    # Calculate area (in physical units)
    pixel_area = (img_dict.get('width', 1) / img.shape[1]) * (img_dict.get('height', 1) / img.shape[0])
    area = np.sum(mask) * pixel_area
    
    # Calculate max height and pit depth relative to average
    if len(segment_data) > 0:
        max_height = (np.max(segment_data) - avg_height) * scale_factor
        pit_depth = (avg_height - np.min(segment_data)) * scale_factor
    else:
        max_height = 0
        pit_depth = 0
    
    # Calculate slope (gradient magnitude)
    if np.sum(mask) > 1:
        # Create masked gradient
        gy, gx = np.gradient(img)
        gx_masked = gx[mask]
        gy_masked = gy[mask]
        
        # Calculate average gradient magnitude (slope)
        avg_slope = np.mean(np.sqrt(gx_masked**2 + gy_masked**2)) * scale_factor
    else:
        avg_slope = 0
    
    return {
        'avg_height': avg_height,
        'rms_roughness': rms_roughness,
        'mean_roughness': mean_roughness,
        'skewness': skew,
        'kurtosis': kurtosis,
        'area': area,
        'avg_slope': avg_slope,
        'max_height': max_height,
        'pit_depth': pit_depth,
        'unit': prefix + unit,
        'area_unit': f'{img_dict.get("widthUnit", "um")}²'
    }

def AnalyzeSegments(img_dict, seg_result):
    """
    Analyze segments of an AFM image and return statistics.
    
    Parameters:
    -----------
    img_dict : dict
        Dictionary containing the AFM image data
    seg_result : dict
        Dictionary containing segmentation results
    
    Returns:
    --------
    list : List of dictionaries with segment statistics
    """
    
    # Check if img_dict is a dictionary and extract the image array
    # Extract the image array
    if isinstance(img_dict, dict) and 'img' in img_dict:
        img_array = img_dict['img']
    else:
        img_array = img_dict

    # Get unit info for display
    unit = img_dict.get('unit', 'um')
    if unit == 'um':
        img_array = img_array * 1e-6  # Convert to meters
        display_unit = 'm'
    else:
        display_unit = unit
    scale,prefix = SetPrefix(img_array.ravel())
    display_unit = prefix + display_unit
         

    # Get segment data
    region_masks = seg_result['region_masks']
    # Analyze each segment
    segment_stats = []
    for i, mask in enumerate(region_masks):
        stats = analyze_segment(img_array*scale, mask, img_dict)
        stats['region_id'] = i + 1  # 1-indexed region IDs
        segment_stats.append(stats)

    # Create a table with the segment statistics using pandas
    
    # Prepare data for the DataFrame
    data = {
        'Region': [stat['region_id'] for stat in segment_stats],
        f'Avg Height ({display_unit})': [stat['avg_height'] for stat in segment_stats],
        f'RMS Rough ({display_unit})': [stat['rms_roughness'] for stat in segment_stats],
        f'Mean Rough ({display_unit})': [stat['mean_roughness'] for stat in segment_stats],
        'Skewness': [stat['skewness'] for stat in segment_stats],
        'Kurtosis': [stat['kurtosis'] for stat in segment_stats],
        f'Area ({img_dict.get("widthUnit", "um")}²)': [stat['area'] for stat in segment_stats],
        f'Avg Slope ({display_unit})': [stat['avg_slope'] for stat in segment_stats],
        f'Max Height ({display_unit})': [stat['max_height'] for stat in segment_stats],
        f'Pit Depth ({display_unit})': [stat['pit_depth'] for stat in segment_stats]
    }
    
    # Create DataFrame
    df_stats = pd.DataFrame(data)
    
    # Format numeric columns to improve readability
    df_stats[f'Avg Height ({display_unit})'] = df_stats[f'Avg Height ({display_unit})'].map('{:.4f}'.format)
    df_stats[f'RMS Rough ({display_unit})'] = df_stats[f'RMS Rough ({display_unit})'].map('{:.4f}'.format)
    df_stats[f'Mean Rough ({display_unit})'] = df_stats[f'Mean Rough ({display_unit})'].map('{:.4f}'.format)
    df_stats['Skewness'] = df_stats['Skewness'].map('{:.2f}'.format)
    df_stats['Kurtosis'] = df_stats['Kurtosis'].map('{:.2f}'.format)
    df_stats[f'Area ({img_dict.get("widthUnit", "um")}²)'] = df_stats[f'Area ({img_dict.get("widthUnit", "um")}²)'].map('{:.2f}'.format)
    df_stats[f'Avg Slope ({display_unit})'] = df_stats[f'Avg Slope ({display_unit})'].map('{:.4f}'.format)
    df_stats[f'Max Height ({display_unit})'] = df_stats[f'Max Height ({display_unit})'].map('{:.4f}'.format)
    df_stats[f'Pit Depth ({display_unit})'] = df_stats[f'Pit Depth ({display_unit})'].map('{:.4f}'.format)
    
    return df_stats
def segment_afm_image_height(image, n_classes=None, min_size=50, morphology_cleanup=True,
                            kernel_scale_factor=0.01, sigma=1.0, auto_classes=True):
    """
    Segment an AFM image using multi-Otsu thresholding.
    
    Parameters:
    -----------
    image : numpy.ndarray or dict
        Input AFM image (2D array) or image dictionary
    n_classes : int or None
        Number of classes (segments) to create. If None and auto_classes is True,
        will automatically determine optimal number of classes.
    min_size : int
        Minimum region size in pixels
    morphology_cleanup : bool
        Apply morphological operations for error correction
    kernel_scale_factor : float
        Scale factor for morphological kernel size based on image resolution
    sigma : float
        Standard deviation for Gaussian filter pre-processing
    auto_classes : bool
        Whether to automatically determine the optimal number of classes
        
    Returns:
    --------
    dict : Dictionary containing segmentation results
    """
    # Extract image array if dictionary is provided
    if isinstance(image, dict) and 'img' in image:
        img = image['img'].copy()
    else:
        img = image.copy()
    
    # Normalize image to 0-1 range
    if img.min() != img.max():
        img_norm = (img - img.min()) / (img.max() - img.min())
    else:
        img_norm = img.copy()
    
    # Apply Gaussian filter to reduce noise
    smoothed = ndimage.gaussian_filter(img_norm, sigma=sigma)
    
    # Automatically determine optimal number of classes if needed
    if auto_classes and n_classes is None:
        # Try different numbers of classes and choose based on variance
        max_classes = min(6, 1 + int(np.log2(len(np.unique(smoothed)))))
        best_variance = 0
        best_classes = 2  # Default to at least 2 classes
        
        for classes in range(2, max_classes + 1):
            try:
                thresholds = threshold_multiotsu(smoothed, classes=classes)
                # Create regions based on thresholds
                regions = np.digitize(smoothed, thresholds)
                
                # Calculate variance ratio (between-class variance / total variance)
                total_var = np.var(smoothed)
                between_var = 0
                
                for i in range(classes):
                    mask = regions == i
                    if np.sum(mask) > 0:
                        between_var += np.sum(mask) * (np.mean(smoothed[mask]) - np.mean(smoothed))**2
                
                between_var /= len(smoothed.ravel())
                variance_ratio = between_var / total_var if total_var > 0 else 0
                
                # If this number of classes provides better separation, use it
                if variance_ratio > best_variance:
                    best_variance = variance_ratio
                    best_classes = classes
                    
                # Stop if we have good separation already
                if variance_ratio > 0.8:
                    break
            except:
                # If multi-Otsu fails for this number of classes, skip it
                continue
        
        n_classes = best_classes
    
    # Default to 3 classes if not specified and auto-detection fails
    if n_classes is None:
        n_classes = 3
    
    # Apply multi-Otsu thresholding
    try:
        thresholds = threshold_multiotsu(smoothed, classes=n_classes)
        regions = np.digitize(smoothed, thresholds)
    except:
        # Fallback to simple thresholding if multi-Otsu fails
        threshold = threshold_otsu(smoothed)
        regions = (smoothed > threshold).astype(int)
    
    # Calculate kernel size based on image resolution
    if isinstance(image, dict) and 'width' in image and 'height' in image:
        rows, cols = img.shape
        # Calculate pixels per unit
        resolution = min(rows / image['height'], cols / image['width'])
        # Scale kernel size based on resolution
        kernel_size = max(2, int(resolution * kernel_scale_factor))
    else:
        # Default kernel size if resolution information is not available
        kernel_size = 3
    
    # Apply morphological operations for error correction if requested
    if morphology_cleanup:
        # Create morphological kernel based on resolution
        morph_kernel = np.ones((kernel_size, kernel_size))
        
        # Process each region separately for cleaner boundaries
        cleaned_regions = np.zeros_like(regions)
        for i in range(n_classes):
            region_mask = (regions == i)
            # Fill small holes
            filled = ndimage.binary_fill_holes(region_mask)
            # Remove small isolated regions
            opened = ndimage.binary_opening(filled, structure=morph_kernel)
            # Add back to the result with the correct label
            cleaned_regions[opened] = i
        
        # If any pixels were lost in cleaning, assign them to nearest region
        if np.any(cleaned_regions == 0) and np.any(cleaned_regions > 0):
            # Distance transform to find nearest non-zero region
            distance_map = ndimage.distance_transform_edt(cleaned_regions == 0, 
                                                            return_distances=False,
                                                            return_indices=True)
            # Assign zero pixels to nearest region
            zero_mask = cleaned_regions == 0
            cleaned_regions[zero_mask] = regions[tuple(distance_map[..., zero_mask])]
        
        regions = cleaned_regions
    
    # Label connected regions (some regions might still be disconnected)
    labeled_image = np.zeros_like(regions, dtype=int)
    next_label = 1
    
    # Process each intensity class
    for i in range(n_classes):
        class_mask = (regions == i)
        # Label connected components in this class
        labels, num = ndimage.label(class_mask)
        
        # Add to final labeling with unique labels
        for j in range(1, num + 1):
            component_mask = (labels == j)
            # Check size
            if np.sum(component_mask) >= min_size:
                labeled_image[component_mask] = next_label
                next_label += 1
    
    # If no regions meet the size criteria, use original regions
    if next_label == 1:
        for i in range(n_classes):
            class_mask = (regions == i)
            if np.sum(class_mask) > 0:
                labeled_image[class_mask] = i + 1
        next_label = n_classes + 1
    
    # Create edge map (boundaries between regions)
    edge_map = ~segmentation.find_boundaries(labeled_image, mode='outer')
    
    # Create masks for each region
    region_masks = []
    for i in range(1, next_label):
        if np.any(labeled_image == i):
            region_masks.append(labeled_image == i)
    
    # Extract region properties
    region_props = measure.regionprops(labeled_image, img)
    
    return {
        'labeled_image': labeled_image,
        'region_masks': region_masks,
        'region_properties': region_props,
        'edge_map': edge_map,
        'thresholds': thresholds if 'thresholds' in locals() else [threshold_otsu(smoothed)]
    }
def segment_afm_image(image, method=EdgeDetectionMethod.Sobel, sigma=1.0, 
                                    threshold=0.2, min_size=50, morphology_cleanup=True, 
                                    kernel_scale_factor=0.01 ):
    """
    Segment an AFM image using a specified edge detection method.
    
    Parameters:
    -----------
    image : numpy.ndarray or dict
        Input AFM image (2D array) or image dictionary
    method : EdgeDetectionMethod
        Edge detection method to use
    sigma : float
        Standard deviation for Gaussian filter pre-processing
    threshold : float
        Threshold for edge detection (0.0 to 1.0)
    min_size : int
        Minimum region size in pixels
    morphology_cleanup : bool
        Apply morphological operations for error correction
    kernel_scale_factor : float
        Scale factor for morphological kernel size based on image resolution
     
    Returns:
    --------
    dict : Dictionary containing segmentation results
    """
    # Extract image array if dictionary is provided
    if isinstance(image, dict) and 'img' in image:
        img = image['img'].copy()
    else:
        img = image.copy()
    
    # Normalize image to 0-1 range
    if img.min() != img.max():
        img_norm = (img - img.min()) / (img.max() - img.min())
    else:
        img_norm = img.copy()
    
    # Apply Gaussian filter to reduce noise
    smoothed = ndimage.gaussian_filter(img_norm, sigma=sigma)
    
    # Apply edge detection based on selected method
    if method == EdgeDetectionMethod.Prewitt:
        grad_x = ndimage.prewitt(smoothed, axis=1)
        grad_y = ndimage.prewitt(smoothed, axis=0)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    elif method == EdgeDetectionMethod.Roberts:
        grad_x = ndimage.roberts(smoothed, axis=1)
        grad_y = ndimage.roberts(smoothed, axis=0)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    elif method == EdgeDetectionMethod.Canny:
        # Calculate Otsu thresholds for Canny parameters
        try:
            thresholds = threshold_multiotsu(smoothed, classes=3)
            low_threshold = thresholds[0]
            high_threshold = thresholds[1]
        except:
            # Fallback if Otsu fails
            low_threshold = threshold * 0.5
            high_threshold = threshold
        
        # Apply Canny edge detection
        edges = canny(smoothed, sigma=sigma, low_threshold=low_threshold, 
                        high_threshold=high_threshold)
        
        # Invert since Canny returns edges as True
        binary_edges = ~edges
    
    else:  # Default to Sobel
        grad_x = ndimage.sobel(smoothed, axis=1)
        grad_y = ndimage.sobel(smoothed, axis=0)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create binary edge map (except for Canny which already creates binary output)
    if method != EdgeDetectionMethod.Canny:
        # Normalize edge magnitude
        if edge_magnitude.max() > 0:
            edge_magnitude = edge_magnitude / edge_magnitude.max()
        
        # Threshold to get binary edges
        binary_edges = edge_magnitude < threshold
    
    # Calculate kernel size based on image resolution
    if isinstance(image, dict) and 'width' in image and 'height' in image:
        rows, cols = img.shape
        # Calculate pixels per unit
        resolution = min(rows / image['height'], cols / image['width'])
        # Scale kernel size based on resolution
        kernel_size = max(2, int(resolution * kernel_scale_factor))
    else:
        # Default kernel size if resolution information is not available
        kernel_size = 3
    
    # Apply morphological operations for error correction if requested
    if morphology_cleanup:
        # Create morphological kernel based on resolution
        morph_kernel = np.ones((kernel_size, kernel_size))
        
        # Close small gaps in edges
        binary_edges = ndimage.binary_closing(binary_edges, structure=morph_kernel)
        
        # Remove small isolated edge fragments
        binary_edges = ndimage.binary_opening(binary_edges, structure=np.ones((max(2, kernel_size//2), max(2, kernel_size//2))))
        
        # Fill small holes in regions
        binary_edges = ndimage.binary_fill_holes(binary_edges)
    
    # Label connected regions
    labeled_image, _ = ndimage.label(binary_edges)
    
    # Remove small regions
    region_sizes = np.bincount(labeled_image.ravel())
    mask_sizes = region_sizes >= min_size
    mask_sizes[0] = 0  # Remove background
    cleaned_labels = mask_sizes[labeled_image]
    
    # Relabel regions after cleaning
    labeled_cleaned, num_cleaned = ndimage.label(cleaned_labels)
    
    # Extract region properties
    region_props = measure.regionprops(labeled_cleaned, img)
    
    # Create masks for each region
    region_masks = []
    for i in range(1, num_cleaned + 1):
        region_masks.append(labeled_cleaned == i)



    
    return {
        'labeled_image': labeled_cleaned,
        'region_masks': region_masks,
        'region_properties': region_props,
        'edge_map': binary_edges  # Invert to get edges as True
    }        



import numpy as np    
from MathTools import SetPrefix
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.spatial import distance
from skimage.filters import threshold_otsu
 
def particle_analysis(image, min_distance=10, min_height_percentile=80, threshold_method='percentile',
                        min_size=5, max_size=1000, segment_regions=None, distance_metrics=True):
    """
    Analyze particles in AFM images using watershed segmentation.
    
    Parameters:
    -----------
    image : dict or np.ndarray
        AFM image data or dictionary containing image data
    min_distance : int
        Minimum number of pixels separating peaks for watershed segmentation
    min_height_percentile : int
        Height percentile threshold for particle detection (0-100), used if threshold_method='percentile'
    threshold_method : str
        Method for thresholding: 'percentile' or 'otsu'
    min_size : int
        Minimum particle size in pixels
    max_size : int
        Maximum particle size in pixels
    segment_regions : list or None
        List of segment masks for region-specific analysis (from segment_afm_image function)
    distance_metrics : bool
        Whether to calculate interparticle distances (can be slow for large particle counts)
    
    Returns:
    --------
    dict: Dictionary containing:
        - 'data': pandas.DataFrame with particle analysis results
        - 'mask': Binary mask showing particle locations
        - 'locations': List of particle centroids (row, col) in pixel coordinates
    """
    # Extract image data and metadata
    if isinstance(image, dict) and 'img' in image:
        img_data = image['img'].copy()
        width = image.get('width', 1.0)
        height = image.get('height', 1.0)
        width_unit = image.get('widthUnit', 'um')
        height_unit = image.get('heightUnit', 'um')
        z_unit = image.get('unit', 'um')
    else:
        img_data = image.copy()
        width = 1.0
        height = 1.0
        width_unit = 'um'
        height_unit = 'um'
        z_unit = 'um'
    
    # Convert to SI units (meters) for internal calculations
    unit_factors = {'m': 1.0, 'um': 1e-6, 'nm': 1e-9, 'pm': 1e-12, 'fm': 1e-15}
    
    width_m = width * unit_factors.get(width_unit, 1.0)
    height_m = height * unit_factors.get(height_unit, 1.0)
    
    # Calculate pixel dimensions
    rows, cols = img_data.shape
    pixel_width_m = width_m / cols
    pixel_height_m = height_m / rows
    pixel_area_m2 = pixel_width_m * pixel_height_m
    
    # Convert z values to meters
    z_factor = unit_factors.get(z_unit, 1.0)
    img_data_m = img_data * z_factor
    
   
    
    # Prepare regions for analysis
    if segment_regions is not None and isinstance(segment_regions, dict) and 'region_masks' in segment_regions:
        regions = segment_regions['region_masks']
        region_labels = [f"region_{i+1}" for i in range(len(regions))]
    elif segment_regions is not None and isinstance(segment_regions, list):
        regions = segment_regions
        region_labels = [f"region_{i+1}" for i in range(len(regions))]
    else:
        # Use the entire image as a single region
        regions = [np.ones_like(img_data, dtype=bool)]
        region_labels = ["whole_image"]
    
    # Initialize results container
    all_results = []
    
    # Create a mask for all particles
    particle_mask = np.zeros_like(img_data, dtype=bool)
    all_particle_locations = []
    
    # Process each region
    for region_idx, region_mask in enumerate(regions):
        # Apply region mask
        masked_data = img_data_m.copy()
        if not np.all(region_mask):
            masked_data[~region_mask] = np.min(masked_data)
        
        # Calculate height threshold for peak detection
        if threshold_method.lower() == 'otsu':
            # Normalize data for Otsu to work better
            norm_data = masked_data[region_mask]
            norm_data = (norm_data - np.min(norm_data)) / (np.max(norm_data) - np.min(norm_data))
            otsu_threshold = threshold_otsu(norm_data)
            height_threshold = np.min(masked_data[region_mask]) + otsu_threshold * (np.max(masked_data[region_mask]) - np.min(masked_data[region_mask]))
        else:  # default to percentile
            height_threshold = np.percentile(masked_data[region_mask], min_height_percentile)
        
        # Find local maxima (particle peaks)
        binary_peaks = masked_data > height_threshold
        distance_array = ndi.distance_transform_edt(binary_peaks)
        
        # Find local maxima with minimum separation
        coordinates = peak_local_max(distance_array, min_distance=min_distance, 
                                        footprint=np.ones((3, 3)), labels=region_mask)
        
        # If no peaks found, skip this region
        if len(coordinates) == 0:
            all_results.append({
                'segment_id': region_labels[region_idx],
                'num_particles': 0,
                'avg_size (nm²)': 0.0,
                'avg_height (nm)': 0.0,
                'std_size (nm²)': 0.0,
                'std_height (nm)': 0.0,
                'avg_interparticle_distance (nm)': 0.0,
                'std_interparticle_distance (nm)': 0.0
            })
            continue
        
        # Prepare markers for watershed
        markers = np.zeros_like(masked_data, dtype=int)
        for i, (x, y) in enumerate(coordinates):
            markers[x, y] = i + 1
        
        # Apply watershed to segment particles
        elevation_map = -distance_array
        labels = watershed(elevation_map, markers, mask=region_mask)
        
        # Extract and analyze each particle
        particle_sizes = []
        particle_heights = []
        particle_volumes = []
        particle_centroids = []
        particle_locations = []
        region_particle_mask = np.zeros_like(img_data, dtype=bool)
        
        for i in range(1, len(coordinates) + 1):
            # Get particle mask
            particle = labels == i
            size_pixels = np.sum(particle)
            
            # Filter by size
            if size_pixels < min_size or size_pixels > max_size:
                continue
            
            # Calculate particle properties
            particle_data = masked_data[particle]
            
            if len(particle_data) == 0:
                continue
                
            # Calculate size in physical units
            size_m2 = size_pixels * pixel_area_m2
            
            # Calculate height statistics relative to local base level
            local_base = np.percentile(particle_data, 10)  # Use 10th percentile as base
            particle_height = np.max(particle_data) - local_base
            
            # Calculate volume by summing heights above local base
            volume_m3 = np.sum(particle_data - local_base) * pixel_area_m2
            volume_m3 = max(0, volume_m3)  # Ensure non-negative volume
            
            # Append particle measurements
            particle_sizes.append(size_m2)
            particle_heights.append(particle_height)
            particle_volumes.append(volume_m3)
            
            # Add to the particle mask
            region_particle_mask = np.logical_or(region_particle_mask, particle)
            
            # Get centroid for distance calculations
            if np.any(particle):
                y_indices, x_indices = np.where(particle)
                centroid_y = np.mean(y_indices)
                centroid_x = np.mean(x_indices)
                particle_centroids.append((centroid_x, centroid_y))
                particle_locations.append((int(centroid_y), int(centroid_x)))  # Row, col format (y, x)
        
        # Calculate inter-particle distances if required
        interparticle_distances = []
        if distance_metrics and len(particle_centroids) > 1:
            for i in range(len(particle_centroids)):
                for j in range(i+1, len(particle_centroids)):
                    dist = distance.euclidean(
                        (particle_centroids[i][0] * pixel_width_m, particle_centroids[i][1] * pixel_height_m),
                        (particle_centroids[j][0] * pixel_width_m, particle_centroids[j][1] * pixel_height_m)
                    )
                    interparticle_distances.append(dist)
        
        # Update the global particle mask and locations
        particle_mask = np.logical_or(particle_mask, region_particle_mask)
        all_particle_locations.extend(particle_locations)
        
        # Convert to arrays for statistics
        particle_sizes = np.array(particle_sizes)
        particle_heights = np.array(particle_heights)
        interparticle_distances = np.array(interparticle_distances)
        
        # Default to nanometer scale for AFM particles
        scale_size, prefix_size = SetPrefix(particle_sizes * 1e18)  # Convert to nm²
        prefix_size = 'n'  # Force nanometer units for typical AFM particles
        scale_size = 1e18  # m² to nm²
        
        scale_height, prefix_height = SetPrefix(particle_heights * 1e9)  # Convert to nm
        prefix_height = 'n'  # Force nanometer units
        scale_height = 1e9  # m to nm
        
        if len(interparticle_distances) > 0:
            scale_dist, prefix_dist = SetPrefix(interparticle_distances * 1e9)  # Convert to nm
            prefix_dist = 'n'  # Force nanometer units
            scale_dist = 1e9  # m to nm
            avg_distance = np.mean(interparticle_distances) * scale_dist
            std_distance = np.std(interparticle_distances) * scale_dist if len(interparticle_distances) > 1 else 0.0
        else:
            prefix_dist = 'n'
            scale_dist = 1e9
            avg_distance = 0.0
            std_distance = 0.0
        
        # Store results for this region
        region_result = {
            'segment_id': region_labels[region_idx],
            'num_particles': len(particle_sizes),
            f'avg_size ({prefix_size}m²)': np.mean(particle_sizes) * scale_size if len(particle_sizes) > 0 else 0.0,
            f'avg_height ({prefix_height}m)': np.mean(particle_heights) * scale_height if len(particle_heights) > 0 else 0.0,
            f'std_size ({prefix_size}m²)': np.std(particle_sizes) * scale_size if len(particle_sizes) > 1 else 0.0,
            f'std_height ({prefix_height}m)': np.std(particle_heights) * scale_height if len(particle_heights) > 1 else 0.0,
            f'avg_interparticle_distance ({prefix_dist}m)': avg_distance,
            f'std_interparticle_distance ({prefix_dist}m)': std_distance,
        }
        
        all_results.append(region_result)
    
    # Return results as dictionary with DataFrame, mask and particle locations
    return {
        'data': pd.DataFrame(all_results),
        'mask': particle_mask,
        'locations': all_particle_locations
    }

def fast_flatten_afm_image(topography):
    """
    Optimized AFM image flattening using main peak analysis.
    
    Parameters:
    -----------
    topography : 2D numpy array
        Height data from AFM measurement
    
    Returns:
    --------
    background : 2D numpy array
        Fitted polynomial background
    n_peaks : int
        Number of detected terraces
    flattened : 2D numpy array
        Flattened topography data
    """
    # Normalize data to improve optimization stability
    topography = topography - np.mean(topography)
    scale = np.std(topography)
    topography = topography / scale
    
    # Initial guess: planar fit using mean gradients
    gy, gx = np.gradient(topography)
    initial_coeffs = np.zeros(10)
   
    
    # Fast optimization with reduced iterations
    result = minimize(fast_loss_function, initial_coeffs, 
                     args=(topography,),
                     method='Powell',  # Powell method is often faster for this type of problem
                     options={'maxiter': 100, 'ftol': 1e-3})
    
    # Generate background and flatten
    background = fast_polynomial_background(topography.shape, result.x)
    flattened = topography - background
    
    # Count peaks using histogram
    hist, _ = np.histogram(flattened, bins=50)
    peaks = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0] + 1
    n_peaks = len(peaks)
    
    # Rescale back to original units
    background = background * scale
    flattened = flattened * scale
    
    return background, n_peaks, flattened-np.min(flattened)


def terrace_flatten_afm_image(topography):
    """
    Optimized AFM image flattening using main peak analysis.
    
    Parameters:
    -----------
    topography : 2D numpy array
        Height data from AFM measurement
    
    Returns:
    --------
    background : 2D numpy array
        Fitted polynomial background
    n_peaks : int
        Number of detected terraces
    flattened : 2D numpy array
        Flattened topography data
    """
    # Normalize data to improve optimization stability
    topography = topography - np.mean(topography)
    scale = np.std(topography)
    topography = topography / scale
    
    # Initial guess: planar fit using mean gradients
    initial_coeffs = np.zeros(10)
    
    # Fast optimization with reduced iterations
    result = minimize(peakH_loss_function, initial_coeffs, 
                     args=(topography,),
                     method='Powell',  # Powell method is often faster for this type of problem
                     options={'maxiter': 100, 'ftol': 1e-3})
    
    # Generate background and flatten
    background = MI.fast_polynomial_background(topography.shape, result.x)
    flattened = topography - background
    
    # Count peaks using histogram
    hist, _ = np.histogram(flattened, bins=50)
    peaks = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0] + 1
    n_peaks = len(peaks)
    
    # Rescale back to original units
    background = background * scale
    flattened = flattened * scale
    
    return background, n_peaks, flattened-np.min(flattened)
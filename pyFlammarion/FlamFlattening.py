
import numpy as np
from enum import Enum
from scipy.stats import trim_mean
from scipy.optimize import minimize
from skimage.filters import threshold_multiotsu
from scipy import ndimage
from sklearn.linear_model import RANSACRegressor
from FileLoaders.FlammarionFile import FlammarionFile, FlammarionImageData
from scipy.signal import find_peaks

class AFMFlatteningMethod(Enum):
    MedianLine = 1 
    MedianOfDifferenceLine = 2
    ModusLine = 3 
    MatchingLine = 4 
    FacetLevelTiltPlane = 5 
    PolynomialLine = 6
    TrimmedMeanLine = 7
    TrimmedMeanOfDifferenceLine = 8 
    TerraceLine = 9        
    PlaneLevel=10 
    ThreePointPlane=11
    polynomialPlaneFlattening= 12
    TerracePlanes= 13 
    Zero = 14
    ZeroFloor = 15

class RegressionMethod(Enum):
    """
    Enum for regression methods.
    """
    LEAST_SQUARES = 'least_squares'
    ABSOLUTE_DEVIATION = 'absolute_deviation'
    RANSAC = 'ransac'

def _trimmed_mean_lineFlattening(imageData: FlammarionImageData, trim_ratio=0.1, mask:np.array=None):
    """
    Perform trimmed mean of difference flattening on an AFM image.

    Args:
        imageData (FlammarionImageData): Image with topography data.
        trim_ratio (float): Fraction to trim from each end of the data when computing the mean.
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.

    Returns:
        np.ndarray: Flattened topography image.
    """
    
    topography = imageData.data
    # Get image dimensions
    rows, _ = topography.shape
    
    
    # Process each row by subtracting the trimmed mean
    for i in range(rows):
        row_data = topography[i, :]
        
        # Apply mask if provided
        if mask is not None:
            row_mask = mask[i, :]
            valid_data = row_data[row_mask == 1]
            # If no valid points in this row, use all points
            if len(valid_data) == 0:
                valid_data = row_data
        else:
            valid_data = row_data
            
        # Calculate trimmed mean for each row using only valid data
        row_mean = trim_mean(valid_data, trim_ratio)
        
        # Subtract the mean from each row
        topography[i, :] = row_data - row_mean
    
    imageData.processingHistory.append(f"Trimmed mean line flattening with trim ratio {trim_ratio}")
    return imageData            

def _trimmed_mean_of_difference_lineFlattening(imageData: FlammarionImageData, trim_ratio=0.1, mask:np.array=None):
    """
    Perform trimmed mean of difference flattening on an AFM image.
    
    Args:
        imageData (FlammarionImageData): Image with topography data (can be image dict or direct array)
        trim_ratio (float): Fraction to trim from each end of the data when computing the mean.
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        
    Returns:
        dict or np.ndarray: Flattened topography image in the same format as input
    """
     
    img = imageData.data
    
    # Get image dimensions
    rows, _ = img.shape
    
    # Process each row
    for i in range(rows-1):
        # Calculate row differences
        differences = img[i+1, :] - img[i, :]
        
        # Apply mask if provided
        if mask is not None:
            # Use only points that are valid in both rows
            combined_mask = (mask[i, :] == 1) & (mask[i+1, :] == 1)
            valid_differences = differences[combined_mask]
            # If no valid points, use all differences
            if len(valid_differences) == 0:
                valid_differences = differences
        else:
            valid_differences = differences
            
        # Calculate trimmed mean of differences
        diff_mean = trim_mean(valid_differences, trim_ratio)
        
        # Adjust the next row based on the differences
        img[i+1, :] = img[i+1, :] - diff_mean
    
    imageData.processingHistory.append(f"Trimmed mean of difference line flattening with trim ratio {trim_ratio}")
    return imageData    
    
def _planeLevelFlattening(imageData: FlammarionImageData, mask:np.array=None, fit_method=RegressionMethod.LEAST_SQUARES, ransac_min_samples=3, 
                        ransac_residual_threshold=0.01, ransac_max_trials=100, absdev_scale=1.0):
    """
    Subtract a fitted plane from an AFM topography image with different fitting methods.
    
    Args:
        imageData (FlammarionImageData): 2D array of topography data
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        fit_method (str): Fitting method: 'least_squares' (MSE), 'absolute_deviation', or 'ransac'
        ransac_min_samples (int): Minimum number of samples for RANSAC fitting
        ransac_residual_threshold (float): Maximum residual for a sample to be considered an inlier in RANSAC
        ransac_max_trials (int): Maximum number of iterations for RANSAC
        absdev_scale (float): Scaling factor for absolute deviation minimization
        
    Returns:
        np.ndarray: Flattened image with plane subtracted
    """
    # Create a copy of the input image
    img = imageData.data
    rows, cols = img.shape
    
    # Create coordinate meshgrid
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Flatten the coordinates and image for fitting
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = img.flatten()
    
    # Create design matrix for plane fit: z = ax + by + c
    A = np.column_stack((x_flat, y_flat, np.ones_like(x_flat)))
    
    # If mask is provided, use it to select good points
    if mask is not None:
        mask_flat = mask.flatten()
        A_masked = A[mask_flat == 1]
        z_masked = z_flat[mask_flat == 1]
    else:
        A_masked = A
        z_masked = z_flat
    
    # Different fitting methods
    if fit_method == RegressionMethod.LEAST_SQUARES:
        # Standard least squares (minimize squared error)
        coeffs, _, _, _ = np.linalg.lstsq(A_masked, z_masked, rcond=None)
        imageData.processingHistory.append(f"Least squares plane fitting")
    
    elif fit_method == RegressionMethod.ABSOLUTE_DEVIATION:
        # Minimize absolute deviation (more robust to outliers)
        def abs_deviation(params):
            a, b, c = params
            plane_values = a * A_masked[:, 0] + b * A_masked[:, 1] + c
            return np.sum(np.abs(z_masked - plane_values)) * absdev_scale
        
        # Initial guess from least squares
        initial_guess, _, _, _ = np.linalg.lstsq(A_masked, z_masked, rcond=None)
        result = minimize(abs_deviation, initial_guess, method='Nelder-Mead')
        coeffs = result.x
        imageData.processingHistory.append(f"Absolute deviation plane fitting")
    elif fit_method ==  RegressionMethod.RANSAC:
        # RANSAC for robust fitting
        
        # Prepare data for scikit-learn
        X = A_masked[:, :2]  # x and y coordinates
        Y = z_masked
        
        # Create and fit RANSAC model
        ransac = RANSACRegressor(
            min_samples=ransac_min_samples,
            residual_threshold=ransac_residual_threshold,
            max_trials=ransac_max_trials,
            random_state=42
        )
        ransac.fit(X, Y)
        
        # Extract coefficients
        a, b = ransac.estimator_.coef_
        c = ransac.estimator_.intercept_
        coeffs = [a, b, c]
        imageData.processingHistory.append(f"RANSAC plane fitting")
    else:
        raise ValueError(f"Unknown fit_method: {fit_method}. Use 'least_squares', 'absolute_deviation', or 'ransac'.")
    
    # Create the plane using the fitted coefficients
    a, b, c = coeffs
    plane = a * x + b * y + c
    
    # Subtract the plane from the image
    imageData.data = img - plane
    
    return imageData
 
def _polynomialPlaneFlattening(imageData: FlammarionImageData, xdegree=3, ydegree=3, mask:np.array=None, fit_method=RegressionMethod.LEAST_SQUARES,
                                ransac_min_samples=None, ransac_residual_threshold=0.01, 
                                ransac_max_trials=100, absdev_scale=1.0):
    """
    Subtract a polynomial surface from an AFM topography image with different fitting methods.
    
    Args:
        imageData (FlammarionImageData): 2D array of topography data
        xdegree (int): Degree of polynomial in x direction (max 4)
        ydegree (int): Degree of polynomial in y direction (max 4)
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        fit_method (RegressionMethod): Fitting method: LEAST_SQUARES, ABSOLUTE_DEVIATION, or RANSAC
        ransac_min_samples (int, optional): Minimum number of samples for RANSAC fitting. If None, calculated based on polynomial terms.
        ransac_residual_threshold (float): Maximum residual for a sample to be considered an inlier in RANSAC
        ransac_max_trials (int): Maximum number of iterations for RANSAC
        absdev_scale (float): Scaling factor for absolute deviation minimization
        
    Returns:
        np.ndarray: Flattened image with polynomial surface subtracted
    """
    # Limit polynomial degrees to prevent instability
    xdegree = min(xdegree, 4)
    ydegree = min(ydegree, 4)
    
    # Create a copy of the input image
    img = imageData.data
    rows, cols = img.shape
    
    # Create coordinate meshgrid
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    
    # Normalize coordinates to [0,1] to improve numerical stability
    x_norm = x / (cols - 1)
    y_norm = y / (rows - 1)
    
    # Flatten arrays for fitting
    x_flat = x_norm.flatten()
    y_flat = y_norm.flatten()
    z_flat = img.flatten()
    
    # Create design matrix for polynomial fit: z = sum(a_ij * x^i * y^j)
    # Build list of all polynomial terms
    X_poly = []
    for i in range(xdegree + 1):
        for j in range(ydegree + 1):
            if i + j <= max(xdegree, ydegree):  # Only include cross-terms up to max degree
                X_poly.append((x_flat ** i) * (y_flat ** j))
    
    # Convert to design matrix
    A = np.column_stack(X_poly)
    
    # If mask is provided, use it to select good points
    if mask is not None:
        mask_flat = mask.flatten()
        A_masked = A[mask_flat == 1]
        z_masked = z_flat[mask_flat == 1]
    else:
        A_masked = A
        z_masked = z_flat
    
    # Set default RANSAC min_samples if not provided
    if ransac_min_samples is None:
        ransac_min_samples = A.shape[1] + 1  # Number of polynomial terms + 1
    
    # Different fitting methods
    if fit_method == RegressionMethod.LEAST_SQUARES:
        # Standard least squares solution
        coeffs, _, _, _ = np.linalg.lstsq(A_masked, z_masked, rcond=None)
        imageData.processingHistory.append(f"Least squares polynomial plane fitting")
    elif fit_method == RegressionMethod.ABSOLUTE_DEVIATION:
        # Minimize absolute deviation (more robust to outliers)
        def abs_deviation(params):
            polynomial_values = A_masked @ params
            return np.sum(np.abs(z_masked - polynomial_values)) * absdev_scale
        
        # Initial guess from least squares
        initial_guess, _, _, _ = np.linalg.lstsq(A_masked, z_masked, rcond=None)
        result = minimize(abs_deviation, initial_guess, method='Nelder-Mead')
        coeffs = result.x
        imageData.processingHistory.append(f"Absolute deviation polynomial plane fitting")
    elif fit_method == RegressionMethod.RANSAC:
        # RANSAC for robust fitting
        ransac = RANSACRegressor(
            min_samples=min(ransac_min_samples, len(A_masked)-1),  # Ensure min_samples doesn't exceed data points
            residual_threshold=ransac_residual_threshold,
            max_trials=ransac_max_trials,
            random_state=42
        )
        
        try:
            ransac.fit(A_masked, z_masked)
            coeffs = ransac.estimator_.coef_
            # Add intercept to coefficients
            if hasattr(ransac.estimator_, 'intercept_'):
                coeffs = np.concatenate([[ransac.estimator_.intercept_], coeffs])
            imageData.processingHistory.append(f" RANSAC polynomial plane fitting")
        except Exception as e:
            # Fallback to least squares if RANSAC fails
            print(f"RANSAC failed, falling back to least squares: {e}")
            coeffs, _, _, _ = np.linalg.lstsq(A_masked, z_masked, rcond=None)
            imageData.processingHistory.append(f"Least squares polynomial plane fitting (fallback)")
        
    else:
        raise ValueError(f"Unknown fit_method: {fit_method}. Use RegressionMethod enum.")
    
    # Calculate the polynomial surface using the fitted coefficients
    polynomial_surface = np.zeros_like(img)
    
    # Loop to compute polynomial values at each point
    term_idx = 0
    for i in range(xdegree + 1):
        for j in range(ydegree + 1):
            if i + j <= max(xdegree, ydegree):  # Only include cross-terms up to max degree
                polynomial_surface += coeffs[term_idx] * (x_norm ** i) * (y_norm ** j)
                term_idx += 1
    
    # Subtract the polynomial surface from the image
    imageData.data = img - polynomial_surface
    
    return imageData

def _median_lineFlattening(imageData: FlammarionImageData, mask:np.array=None):
    """
    Perform median line flattening on an AFM topography image.
    
    Args:
        imageData (FlammarionImageData): 2D array of topography data
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        
    Returns:
        np.ndarray: Flattened image with median of each row subtracted
    """
    # Create a copy of the input image
    img = imageData.data
    rows, _ = img.shape
    
    # Process each row by subtracting the median
    for i in range(rows):
        # Use mask if provided
        if mask is not None:
            row_mask = mask[i, :]
            valid_data = img[i, row_mask == 1]
            # If no valid points in this row, use all points
            if len(valid_data) == 0:
                valid_data = img[i, :]
        else:
            valid_data = img[i, :]
            
        row_median = np.median(valid_data)
        img[i, :] -= row_median
    imageData.processingHistory.append(f"Median line flattening")
    return imageData
        
def _modus_lineFlattening(imageData: FlammarionImageData, mask:np.array=None):
    """
    Perform modus (mode) flattening on an AFM topography image.
    
    Args:
        imageData (FlammarionImageData): 2D array of topography data
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        
    Returns:
        np.ndarray: Flattened image with mode of each row subtracted
    """
    # Create a copy of the input image
    img =  imageData.data
    rows, _ = img.shape
    
    # Process each row by subtracting the estimated mode
    for i in range(rows):
        # Use mask if provided
        if mask is not None:
            row_mask = mask[i, :]
            valid_data = img[i, row_mask == 1]
            # If no valid points in this row, use all points
            if len(valid_data) == 0:
                valid_data = img[i, :]
        else:
            valid_data = img[i, :]
            
        # Use a histogram approach to estimate the mode
        hist, bin_edges = np.histogram(valid_data, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        row_mode = bin_centers[np.argmax(hist)]
        img[i, :] -= row_mode
    imageData.processingHistory.append(f"Modus line flattening")
    return imageData

def _median_Of_Difference_lineFlattening(imageData: FlammarionImageData, mask:np.array=None):
    """
    Perform imageData of difference flattening on an AFM topography image.
    
    Args:
        imageData (FlammarionImageData): 2D array of topography data
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        
    Returns:
        np.ndarray: Flattened image using median of differences method
    """
    # Create a copy of the input image
    img = imageData.data
    rows, _ = img.shape
    
    # Process each row based on differences
    for i in range(rows-1):
        # Calculate row differences
        differences = img[i+1, :] - img[i, :]
        
        # Apply mask if provided
        if mask is not None:
            # Use only points that are valid in both rows
            combined_mask = (mask[i, :] == 1) & (mask[i+1, :] == 1)
            valid_differences = differences[combined_mask]
            # If no valid points, use all differences
            if len(valid_differences) == 0:
                valid_differences = differences
        else:
            valid_differences = differences
            
        # Calculate median of differences
        diff_median = np.median(valid_differences)
        
        # Adjust the next row based on the median difference
        img[i+1, :] = img[i+1, :] - diff_median
    imageData.processingHistory.append(f"Median of difference line flattening")
    
    return imageData

def _matching_lineFlattening(imageData: FlammarionImageData, weight_power=2.0, mask:np.array=None):
    """
    Perform imageData flattening on an AFM topography image.
    
    Args:
        imageData (FlammarionImageData): 2D array of topography data
        weight_power (float): Power factor for weighting flat areas more
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        
    Returns:
        np.ndarray: Flattened image using the matching method
    """
    # Create a copy of the input image
    img = imageData.data
    rows, cols = img.shape
    
    # First pass: calculate vertical gradients for weighting
    gradients = np.abs(np.diff(img, axis=0))
    # Add a zero row to match original dimensions
    gradients = np.vstack([gradients, np.zeros((1, cols))])
    
    # Weights favor flat areas (small gradients)
    weights = 1.0 / (gradients + 1e-10)**weight_power
    
    # Apply mask to weights if provided
    if mask is not None:
        weights = weights * mask
    
    # Process each row
    for i in range(1, rows):
        # Calculate weighted differences with mask
        if mask is not None:
            # Use only points that are valid in both rows
            combined_mask = (mask[i-1, :] == 1) & (mask[i, :] == 1)
            # If no valid points, use all points with regular weights
            if np.sum(combined_mask) == 0:
                weighted_diffs = weights[i-1, :] * (img[i, :] - img[i-1, :])
                total_weight = np.sum(weights[i-1, :])
            else:
                # Use only masked points
                row_weights = weights[i-1, :][combined_mask]
                row_diffs = (img[i, :] - img[i-1, :])[combined_mask]
                weighted_diffs = row_weights * row_diffs
                total_weight = np.sum(row_weights)
        else:
            weighted_diffs = weights[i-1, :] * (img[i, :] - img[i-1, :])
            total_weight = np.sum(weights[i-1, :])
        
        # Prevent division by zero
        if total_weight > 0:
            optimal_shift = np.sum(weighted_diffs) / total_weight
        else:
            optimal_shift = 0
            
        # Apply the optimal shift
        img[i, :] -= optimal_shift
    imageData.processingHistory.append(f"Matching line flattening with weight power {weight_power}")
    return imageData

def _polynomial_LineFlattening(imageData: FlammarionImageData, degree=3, mask:np.array=None, fit_method=RegressionMethod.LEAST_SQUARES, 
                            ransac_min_samples=None, ransac_residual_threshold=0.01, 
                            ransac_max_trials=100, absdev_scale=1.0):
    """
    Perform polynomial flattening on an AFM topography image.
    
    Args:
        imageData (FlammarionImageData): 2D array of topography data
        degree (int): Degree of the polynomial fit
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        fit_method (str): Fitting method: 'least_squares' (MSE), 'absolute_deviation', or 'ransac'
        ransac_min_samples (int, optional): Minimum number of samples for RANSAC fitting. If None, defaults to degree+2.
        ransac_residual_threshold (float): Maximum residual for a sample to be considered an inlier in RANSAC
        ransac_max_trials (int): Maximum number of iterations for RANSAC
        absdev_scale (float): Scaling factor for absolute deviation minimization
        
    Returns:
        np.ndarray: Flattened image with polynomial background subtracted
    """
    # Create a copy of the input image
    img = imageData.data
    rows, cols = img.shape
    
    # Create coordinate meshgrid
    x = np.arange(cols)
    
    # Set default RANSAC min_samples if not provided
    if ransac_min_samples is None:
        ransac_min_samples = degree + 2
    
    # Fit and subtract polynomial for each row
    for i in range(rows):
        row_data = img[i, :]
        
        # Apply mask if provided
        if mask is not None:
            row_mask = mask[i, :]
            valid_indices = np.where(row_mask == 1)[0]
            # If no valid points in this row, use all points
            if len(valid_indices) <= degree + 1:  # Need at least degree+1 points for fit
                valid_indices = np.arange(cols)
                valid_x = x
                valid_data = row_data
            else:
                valid_x = x[valid_indices]
                valid_data = row_data[valid_indices]
        else:
            valid_x = x
            valid_data = row_data
            
        # Different fitting methods
        if fit_method == RegressionMethod.LEAST_SQUARES:
            # Standard least squares polynomial fit
            coeffs = np.polyfit(valid_x, valid_data, degree)
            polynomial = np.polyval(coeffs, x)
            imageData.processingHistory.append(f"Least squares polynomial line flattening")
            
        elif fit_method ==  RegressionMethod.ABSOLUTE_DEVIATION:
            # Define function to minimize absolute deviation
            def abs_deviation(params):
                # Calculate polynomial value for each x
                poly_values = np.zeros_like(valid_x, dtype=float)
                for j, coeff in enumerate(params):
                    poly_values += coeff * valid_x**(degree-j)
                
                # Return sum of absolute deviations
                return np.sum(np.abs(valid_data - poly_values)) * absdev_scale
            
            # Initial guess from least squares
            initial_guess = np.polyfit(valid_x, valid_data, degree)
            result = minimize(abs_deviation, initial_guess, method='Nelder-Mead')
            
            # Create polynomial with optimized coefficients
            polynomial = np.zeros_like(x, dtype=float)
            for j, coeff in enumerate(result.x):
                polynomial += coeff * x**(degree-j)
            imageData.processingHistory.append(f"Absolute deviation polynomial line flattening")
        elif fit_method == RegressionMethod.RANSAC:
            # Prepare data for RANSAC
            X = valid_x.reshape(-1, 1)
            y = valid_data
            
            # Create polynomial features
            X_poly = np.vander(valid_x, degree+1)
            
            # RANSAC with polynomial model
            ransac = RANSACRegressor(
                min_samples=min(ransac_min_samples, len(valid_x)-1),  # Ensure min_samples doesn't exceed data points
                residual_threshold=ransac_residual_threshold,
                max_trials=ransac_max_trials,
                random_state=42
            )
            
            try:
                ransac.fit(X_poly, y)
                # Generate the polynomial for all x values
                X_full = np.vander(x, degree+1)
                polynomial = ransac.predict(X_full)
                imageData.processingHistory.append(f"RANSAC polynomial line flattening")
            except:
                # Fallback to least squares if RANSAC fails
                coeffs = np.polyfit(valid_x, valid_data, degree)
                polynomial = np.polyval(coeffs, x)
                imageData.processingHistory.append(f"Least squares polynomial line flattening (fallback)")
        
        else:
            raise ValueError(f"Unknown fit_method: {fit_method}. Use 'least_squares', 'absolute_deviation', or 'ransac'.")
        
        # Subtract the polynomial from the row
        img[i, :] -= polynomial
    
    return imageData

def _three_Point_planeFlattening(imageData: FlammarionImageData, p1, p2, p3, mask:np.array=None):
    """
    Perform three-point plane leveling on an AFM topography image.
    
    Args:
        imageData (FlammarionImageData): 2D array of topography data
        p1, p2, p3: Three points to define the plane
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
            If provided, plane is fit to all good points instead of just 3 points.
        
    Returns:
        np.ndarray: Flattened image with plane defined by three corners subtracted
    """
    img = imageData.data
    rows, cols = img.shape
    
    # If mask is provided, use planeLevelFlattening instead (which supports masks)
    if mask is not None:
        return _planeLevelFlattening(img, mask=mask)
    
    # Otherwise, continue with three-point method
    p1 = (p1[0], p1[1], img[p1[0], p1[1]])
    p2 = (p2[0], p2[1], img[p2[0], p2[1]])
    p3 = (p3[0], p3[1], img[p3[0], p3[1]])
        
    # Calculate plane equation coefficients (Ax + By + Cz + D = 0)
    v1 = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]]
    v2 = [p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]]
    
    # Cross product to find normal vector (A, B, C)
    A = v1[1]*v2[2] - v1[2]*v2[1]
    B = v1[2]*v2[0] - v1[0]*v2[2]
    C = v1[0]*v2[1] - v1[1]*v2[0]
    
    # D coefficient
    D = -(A*p1[0] + B*p1[1] + C*p1[2])
    
    # Create coordinate meshgrid
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Calculate the plane at each point (z = -(Ax + By + D) / C)
    plane = -(A*x + B*y + D) / C if C != 0 else np.zeros_like(img)
    imageData.data = img - plane  # Subtract the plane from the image
    imageData.processingHistory.append(f"Three-point plane flattening with points {p1}, {p2}, {p3}")
    return imageData


def _facet_Level_Tilt_planeFlattening(imageData: FlammarionImageData, mask:np.array=None):
    """
    Perform facet level tilt flattening on an AFM topography image.
    
    Args:
        imageData (FlammarionImageData): 2D array of topography data
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        
    Returns:
        np.ndarray: Flattened image with optimal facet-preserving plane subtracted
    """
    img = imageData.data
    rows, cols = img.shape
    
    # Create coordinate meshgrid
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = img.flatten()
    
    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.flatten()
        valid_mask = mask_flat == 1
        # If no valid points, use all points
        if np.sum(valid_mask) == 0:
            valid_mask = np.ones_like(mask_flat, dtype=bool)
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        z_valid = z_flat[valid_mask]
        # Calculate gradients only on valid regions
        masked_img = np.copy(img)
        masked_img[mask == 0] = np.nan
    else:
        x_valid = x_flat
        y_valid = y_flat
        z_valid = z_flat
        masked_img = img
    
    # Calculate image gradients (ignoring NaNs if mask applied)
    with np.errstate(invalid='ignore'):  # Ignore NaN warnings
        dx = np.gradient(masked_img, axis=1)
        dy = np.gradient(masked_img, axis=0)
    
    # Remove NaNs for histogram
    dx_valid = dx[~np.isnan(dx)]
    dy_valid = dy[~np.isnan(dy)]
    
    # Compute histogram of gradients to identify common facet orientations
    hist_dx, edges_dx = np.histogram(dx_valid, bins=50)
    hist_dy, edges_dy = np.histogram(dy_valid, bins=50)
    
    # Find dominant gradient directions (peaks in histograms)
    peak_dx_idx = np.argmax(hist_dx)
    peak_dy_idx = np.argmax(hist_dy)
    
    # Get bin centers
    centers_dx = (edges_dx[:-1] + edges_dx[1:]) / 2
    centers_dy = (edges_dy[:-1] + edges_dy[1:]) / 2
    
    # Dominant gradients
    dominant_dx = centers_dx[peak_dx_idx]
    dominant_dy = centers_dy[peak_dy_idx]
    
    # Define plane with these gradients
    def plane_error(coeffs):
        a, b, c = coeffs
        # Calculate the plane at valid points
        plane_values = a * x_valid + b * y_valid + c
        # Calculate error between plane and actual heights
        return np.sum((z_valid - plane_values)**2)
    
    # Initial guess based on dominant gradients
    initial_guess = [dominant_dx, dominant_dy, np.nanmedian(masked_img)]
    
    # Optimize to find best plane
    result = minimize(plane_error, initial_guess, method='Nelder-Mead')
    a, b, c = result.x
    
    # Create the plane for all points
    plane = a * x + b * y + c
    
    imageData.data = img - plane  # Subtract the plane from the image
    imageData.processingHistory.append(f"Facet level tilt plane flattening with gradients {dominant_dx}, {dominant_dy}")
    return imageData

def _terrace_lineFlattening(imageData: FlammarionImageData, threshold=None, mask:np.array=None):
    """
    Perform terrace flattening on an AFM topography image by identifying horizontal terraces.
    
    Args:
        imageData (FlammarionImageData): 2D array of topography data
        threshold (float, optional): Threshold for determining terraces, if None it's calculated automatically
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        
    Returns:
        np.ndarray: Flattened image with terrace model subtracted
    """
    
    # Create a copy of the input image
    img = imageData.data
    rows, cols = img.shape
    
    # Apply mask to image for processing if provided
    if mask is not None:
        masked_img = np.copy(img)
        masked_img[mask == 0] = np.nan
    else:
        masked_img = img
    
    # If no threshold is provided, use multiple Otsu thresholding to find terraces
    if threshold is None:
        try:
            # Filter out masked regions for thresholding
            if mask is not None:
                valid_values = img[mask == 1]
            else:
                valid_values = img.ravel()
                
            thresholds = threshold_multiotsu(valid_values, classes=3)
            threshold = thresholds[0]  # Use the first threshold
        except:
            # Fallback if multi-Otsu fails
            if mask is not None:
                threshold = np.nanpercentile(masked_img, 25)
            else:
                threshold = np.percentile(img, 25)
    
    # Identify potential terraces (regions with low local variation)
    local_std = ndimage.generic_filter(img, np.std, size=5, mode='nearest')
    
    # Combine terrace identification with mask if provided
    if mask is not None:
        terrace_mask = (local_std < threshold) & (mask == 1)
    else:
        terrace_mask = local_std < threshold
    
     
    
    # Create meshgrid for all points
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = img.flatten()
    
    # Use only terrace points for fitting if we have enough
    if np.sum(terrace_mask) > 100:
        mask_flat = terrace_mask.flatten()
        A = np.column_stack((x_flat[mask_flat], y_flat[mask_flat], np.ones_like(x_flat[mask_flat])))
        z_terrace = z_flat[mask_flat]
        coeffs, _, _, _ = np.linalg.lstsq(A, z_terrace, rcond=None)
    else:
        # Fallback to using masked points or all points
        if mask is not None and np.sum(mask) > 100:
            mask_flat = mask.flatten()
            A = np.column_stack((x_flat[mask_flat == 1], y_flat[mask_flat == 1], 
                                np.ones_like(x_flat[mask_flat == 1])))
            z_valid = z_flat[mask_flat == 1]
            coeffs, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)
        else:
            # Fallback to using all points
            A = np.column_stack((x_flat, y_flat, np.ones_like(x_flat)))
            coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
    
    # Create the terrace model
    a, b, c = coeffs
    terrace_img = a * x + b * y + c
    
    # Subtract the terrace model from the original image
    imageData.data = img - terrace_img
    imageData.processingHistory.append(f"Terrace line flattening with threshold {threshold}")
    return imageData
    
    
def _terrace_planeFlattening(imageData: FlammarionImageData, mask:np.array=None, sectionPercent=4, polyDegree=3, limitPercent=0.75):
    """
    A method to flatten the AFM image by dividing it into many sections and then using the normal of each 
    section to fit a polynomial plane to the image. This method is useful for images with terraces or steps.
    
    Args:
        image (FlammarionImageData): 2D array of topography data
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        sectionPercent (float): The size of the sections to divide the image into, as a percentage of the image size.
        polyDegree (int): Degree of 2D polynomial to fit (max is 3)
        limitPercent (float): Percentage of the image to use for fitting the polynomial plane, used to avoid edges and spots.
        
    Returns:
        np.ndarray: Flattened image with the polynomial plane subtracted without the offset from terraces.
    """
    image = imageData.data
    rows, cols = image.shape
    sections = int(np.ceil(sectionPercent / 100 * rows))
    
    # Create coordinate meshgrid
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Divide image into sections
    x_sections = y_sections = sections
    x_size = cols // x_sections
    y_size = rows // y_sections
    
    # Initialize array to store plane coefficients for each section
    planeCenters =[]
    slopes = []
    planeError = []
    
    useMask = mask is not None
    
    # Process each section separately
    for i in range(y_sections):
        for j in range(x_sections):
            # Calculate section boundaries
            y_start = i * y_size
            y_end = min((i + 1) * y_size, rows)
            x_start = j * x_size
            x_end = min((j + 1) * x_size, cols)
            # Extract section data
            section = image[y_start:y_end, x_start:x_end]
            if useMask:
                    section_mask = mask[y_start:y_end, x_start:x_end]
                    # Skip sections that are too small or have too few valid points
                    if np.sum(section_mask)/(x_size*y_size) <.3:
                        continue
                    
                    # Use only valid points for plane fitting
                    section_x = x[y_start:y_end, x_start:x_end][section_mask == 1]
                    section_y = y[y_start:y_end, x_start:x_end][section_mask == 1]
                    section_z = section[section_mask == 1]
            else:
                # Use all points in the section
                section_x = x[y_start:y_end, x_start:x_end].flatten()
                section_y = y[y_start:y_end, x_start:x_end].flatten()
                section_z = section.flatten()
            
            # Fit plane to section using least squares
            A = np.column_stack((section_x, section_y, np.ones_like(section_x)))
            coeffs, _, _, _ = np.linalg.lstsq(A, section_z, rcond=None)
            
            a, b, c = coeffs
            # Calculate the plane at valid points
            plane_values = a * section_x + b * section_y + c
            error = np.mean((section_z - plane_values)**2)
            
            planeCenter = (x_start + x_end) / 2, (y_start + y_end) / 2
            planeCenters.append(planeCenter)
            planeError.append(error)
            slopes.append((a, b)) 
    
    # Create output image with sections
    # Filter out sections with high error (outliers)
    sorted_errors = sorted(planeError)
    error_threshold = sorted_errors[int(len(sorted_errors) * limitPercent)]  # Use top 80% best fits
    valid_indices = [i for i, err in enumerate(planeError) if err <= error_threshold]

    # Extract valid plane data
    valid_centers = np.array([planeCenters[i] for i in valid_indices])
    valid_slopes = np.array([slopes[i] for i in valid_indices])

    # If not enough valid sections, return original image
    if len(valid_centers) < 3:
        return image
    
    # Prepare data for polynomial fit
    x_centers = valid_centers[:, 0]
    y_centers = valid_centers[:, 1]

    # Limit polynomial degree to max 3
    polyDegree = min(polyDegree, 3)
    coeffs = np.zeros((polyDegree + 1) * (polyDegree + 2) // 2)

    #valid slopes contain the slopes that we want to fit to the slopes of our background.  
    # We need to solve for coefficients that make the derivatives of our polynomial match the observed slopes
    # The x-derivative of our polynomial should match valid_slopes[:, 0]
    # The y-derivative of our polynomial should match valid_slopes[:, 1]
    
    # Set up the polynomial derivatives for x direction (dx/dy)
    def poly_x_derivative(coeffs, x, y):
        # For a 3rd degree polynomial: f(x,y) = c0 + c1*x + c2*y + c3*x^2 + c4*xy + c5*y^2 + c6*x^3 + c7*x^2*y + c8*x*y^2 + c9*y^3
        # Derivative with respect to x is: df/dx = c1 + 2*c3*x + c4*y + 3*c6*x^2 + 2*c7*x*y + c8*y^2
        if polyDegree >= 3:
            return (coeffs[1] + 
                   2*coeffs[3]*x + coeffs[4]*y + 
                   3*coeffs[6]*x**2 + 2*coeffs[7]*x*y + coeffs[8]*y**2)
        elif polyDegree == 2:
            return coeffs[1] + 2*coeffs[3]*x + coeffs[4]*y
        else:  # Linear
            return coeffs[1]
    
    # Set up the polynomial derivatives for y direction (dx/dy)
    def poly_y_derivative(coeffs, x, y):
        # Derivative with respect to y is: df/dy = c2 + c4*x + 2*c5*y + c7*x^2 + 2*c8*x*y + 3*c9*y^2
        if polyDegree >= 3:
            return (coeffs[2] + 
                   coeffs[4]*x + 2*coeffs[5]*y + 
                   coeffs[7]*x**2 + 2*coeffs[8]*x*y + 3*coeffs[9]*y**2)
        elif polyDegree == 2:
            return coeffs[2] + coeffs[4]*x + 2*coeffs[5]*y
        else:  # Linear
            return coeffs[2]
    
    # Define error function for optimization
    def slope_error(coeffs):
        # Calculate predicted slopes at each center point
        pred_slopes_x = poly_x_derivative(coeffs, x_centers, y_centers)
        pred_slopes_y = poly_y_derivative(coeffs, x_centers, y_centers)
        
        # Calculate error between predicted and observed slopes
        error_x = pred_slopes_x - valid_slopes[:, 0]
        error_y = pred_slopes_y - valid_slopes[:, 1]
        
        # Return sum of squared errors
        return np.sum(error_x**2 + error_y**2)
    
    # Optimize to find the best coefficients
    # Initialize with zeros but appropriate length for the polynomial degree
    num_coeffs = (polyDegree + 1) * (polyDegree + 2) // 2
    initial_guess = np.zeros(num_coeffs)
    
    # Perform the optimization
    result = minimize(slope_error, initial_guess, method='Powell')
    coeffs = result.x
    
    # Create the polynomial background
    # Evaluate polynomial at each point in the image
    plane = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            if polyDegree >= 3:
                plane[i, j] = (coeffs[0] + 
                              coeffs[1]*j + coeffs[2]*i + 
                              coeffs[3]*j**2 + coeffs[4]*j*i + coeffs[5]*i**2 + 
                              coeffs[6]*j**3 + coeffs[7]*j**2*i + coeffs[8]*j*i**2 + coeffs[9]*i**3)
            elif polyDegree == 2:
                plane[i, j] = (coeffs[0] + 
                              coeffs[1]*j + coeffs[2]*i + 
                              coeffs[3]*j**2 + coeffs[4]*j*i + coeffs[5]*i**2)
            else:  # Linear
                plane[i, j] = coeffs[0] + coeffs[1]*j + coeffs[2]*i
    imageData.processingHistory.append(f"TerracePlaneFlattening: Subtracted polynomial plane from image.")
    imageData.data = image - plane
    return imageData
   


def ZeroImage(image: FlammarionImageData, mask:np.array=None ):
        """
        Sets the image zero level to the most common height level (assumed to be the substrate/floor).
        
        Args:
            image (np.ndarray): 2D array of topography data
            mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for analysis
            bins (int): Number of bins for histogram
            min_peak_height (float): Minimum relative height of a peak to be considered
            smoothing (int): Gaussian smoothing sigma for histogram
            
        Returns:
            np.ndarray: Zeroed image with floor level at zero
        """
      

        if not isinstance(image, FlammarionImageData):
            raise TypeError("Input must be a FlammarionImageData object.")
            
        img_array:np.array = image.data     
        # Apply mask if provided
        if mask is not None:
            valid_data = img_array[mask == 1]
            if len(valid_data) == 0:  # If mask excludes all points, use all data
                min = np.min(img_array)
        else:
            min = np.min(img_array)
        
        # Adjust the image so the floor is at zero
        image.data = img_array - min
        image.processingHistory.append(f"ZeroImage: Subtracted {min} from image to set floor level to zero.")
        return image
    
def ZeroImageToFloor(image: FlammarionImageData  , mask:np.array=None)-> FlammarionImageData:
    """
    Sets the image zero level to the most likely floor level by analyzing histogram peaks.
    
    Args:
        image ( FlammarionImageData | FlammarionFile): 2D array of topography data or image dictionary
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for analysis
    Returns:
        FlammarionImageData (edits image in place): Zeroed image with floor level at zero
    """
   
    peak_prominence=0.05
    smoothing=3

    if not isinstance(image, FlammarionImageData):
        raise TypeError("Input must be a FlammarionImageData object.")
        
    img_array:np.array = image.data     
    # Apply mask if provided
    if mask is not None:
        valid_data = img_array[mask == 1]
        if len(valid_data) == 0:  # If mask excludes all points, use all data
            valid_data = img_array.flatten()
    else:
        valid_data = img_array.flatten()
    
    # Create histogram
    hist, bin_edges = np.histogram(valid_data, bins='auto')
    
    
    # Apply smoothing to the histogram
    if smoothing > 0:
        hist_smooth = ndimage.gaussian_filter1d(hist, sigma=smoothing)
    else:
        hist_smooth = hist
    
    hist_max = np.max(hist_smooth)*peak_prominence
    
    # Find peaks in the histogram
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak_indices, _ = find_peaks(hist_smooth, height=hist_max)

    # Check if peaks were found
    if len(peak_indices) == 0:
        return image
    else:
        peak_positions = bin_centers[peak_indices]
        # Find the peak with the lowest height value (not intensity, but actual z-height)
        floor_level = np.min(peak_positions)

    
    # Subtract the floor level from the image data
    image.data = img_array - floor_level
    image.processingHistory.append(f"ZeroImageToFloor: Subtracted {floor_level} from image to set floor level to zero.")
    return image

 
flattening_functions = {
    AFMFlatteningMethod.PlaneLevel: _planeLevelFlattening,
    AFMFlatteningMethod.MedianLine: _median_lineFlattening,
    AFMFlatteningMethod.TrimmedMeanLine: _trimmed_mean_lineFlattening,
    AFMFlatteningMethod.TrimmedMeanOfDifferenceLine: _trimmed_mean_of_difference_lineFlattening,
    AFMFlatteningMethod.MedianOfDifferenceLine: _median_Of_Difference_lineFlattening,
    AFMFlatteningMethod.ModusLine: _modus_lineFlattening,
    AFMFlatteningMethod.MatchingLine: _matching_lineFlattening,
    AFMFlatteningMethod.PolynomialLine: _polynomial_LineFlattening,
    AFMFlatteningMethod.TerraceLine: _terrace_lineFlattening,
    AFMFlatteningMethod.FacetLevelTiltPlane: _facet_Level_Tilt_planeFlattening,
    AFMFlatteningMethod.ThreePointPlane: _three_Point_planeFlattening,
    AFMFlatteningMethod.polynomialPlaneFlattening: _polynomialPlaneFlattening,
    AFMFlatteningMethod.TerracePlanes: _terrace_planeFlattening,
    AFMFlatteningMethod.Zero: ZeroImage,
    AFMFlatteningMethod.ZeroFloor: ZeroImageToFloor
}
 
def FlattenImage(imagePack: FlammarionImageData| FlammarionFile, flattenMethod=AFMFlatteningMethod.PlaneLevel, mask=None, **kwargs):
    """
    Apply flattening to an AFM image using the specified method.
    
    Args:
        imagePack (dict): Dictionary containing the AFM image data
        flattenMethod (AFMFlatteningMethod): Method to use for flattening
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening
        **kwargs: Additional parameters specific to the flattening method
    
    Returns:
        dict: Copy of imagePack with flattened image
    """
    # Ensure mask is passed to the flattening function if provided
    if mask is not None:
        kwargs['mask'] = mask
    if isinstance(imagePack, FlammarionFile):
        for key in imagePack:
            imagePack[key] = FlattenImage(imagePack[key], flattenMethod, mask, **kwargs)
    elif  isinstance(imagePack, FlammarionImageData):
        if flattenMethod in flattening_functions:
            imagePack = flattening_functions[flattenMethod](imagePack, **kwargs)
        else:
            raise ValueError(f"Unknown flattening method: {flattenMethod}")
     
    return imagePack
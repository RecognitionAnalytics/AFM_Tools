
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from scipy.stats import trim_mean
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from matplotlib.colors import Normalize 
from skimage.filters import threshold_multiotsu
from scipy import ndimage
from skimage.feature import canny
from sklearn.linear_model import RANSACRegressor
import pandas as pd
from skimage import filters, segmentation, color, measure
import matplotlib.patches as mpatches
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.spatial import distance
import re
from skimage.segmentation import mark_boundaries
from skimage.filters import threshold_otsu
import  struct 
import os
from array import array
from glob import glob


large = 18
med = 18
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (8, 6),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-v0_8-darkgrid')


def loadMI(filename):
    size=os.path.getsize(filename)
    current=[]
    
    baseTime=os.path.getctime(filename)
    distance=[]
    times=[]
    velocity=[]
    distances=[]
    data = array('B')
    isSpectroscopy=False
    bufferLabels = []
    bufferUnits =[]
    bufferDirections = []
    bufferRange=[]
    displayRange = [] 
    chunkLabels = [] 
    displayOffset = []
    parameters = {}
    with open(filename, "rb") as file1:
        # Reading form a file
        d=file1.readline()
        chunks=[]
        while d!='data          BINARY\n' and d!='data          BINARY_32\n':
            d=file1.readline()
            
            d=d.decode('ascii')
            dd=d.strip().split(' ')
            
            if len(dd)>1:
                ddClean =" ".join( [x for x in dd[1:] if x!=' ']).strip() .replace('FALSE','False').replace('TRUE','True')
                 
                try:
                    ddClean=eval(ddClean)
                except:
                    pass
                
                
                parameters[dd[0]]=ddClean
                
            if len(dd)>1 and dd[-1]=='Spectroscopy':
                isSpectroscopy=True
          
            if dd[0]=='bufferLabel':
                bufferLabels.append(" ".join(dd[1:]).strip())
            if dd[0]=='bufferUnit':
                bufferUnits.append(" ".join(dd[1:]).strip())
            if dd[0]=='direction':
                bufferDirections.append(" ".join(dd[1:]).strip())
            if dd[0]=='bufferRange':
                bufferRange.append(float(dd[-1]))
           
            if dd[0]==('DisplayRange'):
                displayRange.append(float(dd[-1]))
            if dd[0]==('DisplayOffset'):
                displayOffset.append(float(dd[-1]))
                
                    
            if  dd[0]==('bias'):
                bias =float(dd[-1]) 
                
            if  d.startswith('chunk'):
                parts = d.split('\t')
                
                dist=float(parts[5])
                time0=float(parts[2])
                time=float(parts[3])
                valueO=float(parts[4])
                label = parts[-1].strip()
                chunks.append({'points':int(parts[1]),
                               'distperpoint':dist, 
                               'timeperpoint':time,
                               'startValue':valueO,
                               'time0':time0,
                               'label':label})

        

        
        loc = file1.tell()
        binarySize =size-loc
        data.fromfile(file1,binarySize )

        if not isSpectroscopy:
            images = {}
            cc=0
            xPixels = parameters['xPixels']
            yPixels = parameters['yPixels']
            for i in range(0,len(bufferLabels)):
                img = np.zeros((xPixels,yPixels))
                 
                for j in range(0,xPixels):
                    for k in range(0,yPixels):
                        img[j,k] = struct.unpack('i', data[cc:cc+4])[0]
                        cc+=4
                        
                 
                images[bufferLabels[i]+ "_" + bufferDirections[i] ]= ({ 'img':img  * bufferRange[i]/2147483648.0,  
                                'label':bufferLabels[i],
                                'unit':bufferUnits[i],
                                'width':parameters['xLength'] * 1e6,
                                'height':parameters['yLength'] * 1e6,
                                'widthUnit' : 'um',
                                'heightUnit' : 'um',
                                'direction':bufferDirections[i]
                               })
            return {'images':images,'parameters':parameters}
        else:
            i=0
            while True:
                    cDist=0
                   
                    for chunk in chunks:
                        chunkDist=[]
                        chunkCurrent=[]
                        
                        if isSpectroscopy:
                            cDist=chunk['startValue']
                            

                        for j in range(0,chunk['points']):
                            try:
                                f=(struct.unpack('<f', data[i:i+4]))
                            except:
                                print(chunkLabels) 
                                curveChunks = [] 
                                for k in range(len(distance)):
                                    curveChunks.append({
                                        'x':np.array(distance[k]),
                                        'y':np.array(current[k]),
                                        'time':np.array(times[k]),
                                        'velocity':velocity[k],
                                        'label':chunkLabels[k]
                                        }
                                                       )
                                
                                curves =   {
                                        'xlabel':bufferLabels[0],
                                        'xunit':bufferUnits[0],
                                        'ylabel':bufferLabels[1],
                                        'yunit':bufferUnits[1],
                                        'data':curveChunks,
                                        'time':np.array(times[k]),
                                        'velocity':np.array(velocity[k]),
                                    }
                                return {'curves':curves,'parameters':parameters}
                            chunkCurrent.append(f)
                            cDist+=chunk['distperpoint']
                            chunkDist.append(cDist)
                            i+=4
                    
                        dist=np.array(chunkDist)
                        if len(dist)>0:
                            chunkLabels.append(chunk['label'])
                            distance.append(dist)
                            current.append(np.array(chunkCurrent).ravel()) 
                            totalTime=len(chunkDist)*chunk['timeperpoint']
                            time= chunk['time0']+ np.linspace(0, totalTime, len(chunkDist))
                        
                            times.append(time )
                            if len(dist)>0:
                                distances.append((dist[0]-dist[-1]))
                                velocity.append((dist[0]-dist[-1])/(time[0]-time[-1]))


class RegressionMethod(Enum):
    """
    Enum for regression methods.
    """
    LEAST_SQUARES = 'least_squares'
    ABSOLUTE_DEVIATION = 'absolute_deviation'
    RANSAC = 'ransac'

def _trimmed_mean_flattening(topography, trim_ratio=0.1, mask=None):
    """
    Perform trimmed mean of difference flattening on an AFM image.

    Args:
        topography (np.ndarray): Image with topography data.
        trim_ratio (float): Fraction to trim from each end of the data when computing the mean.
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.

    Returns:
        np.ndarray: Flattened topography image.
    """
    
    # Get image dimensions
    rows, _ = topography.shape
    
    # Create an empty array for the flattened image
    flattened_image = np.zeros_like(topography)
    
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
        flattened_image[i, :] = row_data - row_mean
    
    return flattened_image            

def trimmed_mean_of_difference_flattening(topography, trim_ratio=0.1, mask=None):
    """
    Perform trimmed mean of difference flattening on an AFM image.
    
    Args:
        topography (dict or np.ndarray): Image with topography data (can be image dict or direct array)
        trim_ratio (float): Fraction to trim from each end of the data when computing the mean.
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        
    Returns:
        dict or np.ndarray: Flattened topography image in the same format as input
    """
    # Extract the image array if a dictionary is provided
    if isinstance(topography, dict) and 'img' in topography:
        img = topography['img'].copy()
        return_dict = True
    else:
        img = topography.copy()
        return_dict = False
    
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
    
    # Return in the same format as input
    if return_dict:
        topography['img'] = img
        return topography
    else:
        return img
    
def planeLevelFlattening(image, mask=None, fit_method=RegressionMethod.LEAST_SQUARES, ransac_min_samples=3, 
                        ransac_residual_threshold=0.01, ransac_max_trials=100, absdev_scale=1.0):
    """
    Subtract a fitted plane from an AFM topography image with different fitting methods.
    
    Args:
        image (np.ndarray): 2D array of topography data
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
    img = image.copy()
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
    
    else:
        raise ValueError(f"Unknown fit_method: {fit_method}. Use 'least_squares', 'absolute_deviation', or 'ransac'.")
    
    # Create the plane using the fitted coefficients
    a, b, c = coeffs
    plane = a * x + b * y + c
    
    # Subtract the plane from the image
    flattened_img = img - plane
    
    return flattened_img
 
def polynomialPlaneFlattening(image, xdegree=3, ydegree=3, mask=None, fit_method=RegressionMethod.LEAST_SQUARES,
                                ransac_min_samples=None, ransac_residual_threshold=0.01, 
                                ransac_max_trials=100, absdev_scale=1.0):
    """
    Subtract a polynomial surface from an AFM topography image with different fitting methods.
    
    Args:
        image (np.ndarray): 2D array of topography data
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
    img = image.copy()
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
    
    elif fit_method == RegressionMethod.ABSOLUTE_DEVIATION:
        # Minimize absolute deviation (more robust to outliers)
        def abs_deviation(params):
            polynomial_values = A_masked @ params
            return np.sum(np.abs(z_masked - polynomial_values)) * absdev_scale
        
        # Initial guess from least squares
        initial_guess, _, _, _ = np.linalg.lstsq(A_masked, z_masked, rcond=None)
        result = minimize(abs_deviation, initial_guess, method='Nelder-Mead')
        coeffs = result.x
    
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
        except Exception as e:
            # Fallback to least squares if RANSAC fails
            print(f"RANSAC failed, falling back to least squares: {e}")
            coeffs, _, _, _ = np.linalg.lstsq(A_masked, z_masked, rcond=None)
    
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
    flattened_img = img - polynomial_surface
    
    return flattened_img

def medianFlattening(image, mask=None):
    """
    Perform median line flattening on an AFM topography image.
    
    Args:
        image (np.ndarray): 2D array of topography data
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        
    Returns:
        np.ndarray: Flattened image with median of each row subtracted
    """
    # Create a copy of the input image
    img = image.copy()
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
    
    return img
        
def modusFlattening(image, mask=None):
    """
    Perform modus (mode) flattening on an AFM topography image.
    
    Args:
        image (np.ndarray): 2D array of topography data
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        
    Returns:
        np.ndarray: Flattened image with mode of each row subtracted
    """
    # Create a copy of the input image
    img = image.copy()
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
    
    return img

def medianOfDifferenceFlattening(image, mask=None):
    """
    Perform median of difference flattening on an AFM topography image.
    
    Args:
        image (np.ndarray): 2D array of topography data
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        
    Returns:
        np.ndarray: Flattened image using median of differences method
    """
    # Create a copy of the input image
    img = image.copy()
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
    
    return img

def matchingFlattening(image, weight_power=2.0, mask=None):
    """
    Perform matching flattening on an AFM topography image.
    
    Args:
        image (np.ndarray): 2D array of topography data
        weight_power (float): Power factor for weighting flat areas more
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        
    Returns:
        np.ndarray: Flattened image using the matching method
    """
    # Create a copy of the input image
    img = image.copy()
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
    
    return img

def polynomialLineFlattening(image, degree=3, mask=None, fit_method=RegressionMethod.LEAST_SQUARES, 
                            ransac_min_samples=None, ransac_residual_threshold=0.01, 
                            ransac_max_trials=100, absdev_scale=1.0):
    """
    Perform polynomial flattening on an AFM topography image.
    
    Args:
        image (np.ndarray): 2D array of topography data
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
    img = image.copy()
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
            except:
                # Fallback to least squares if RANSAC fails
                coeffs = np.polyfit(valid_x, valid_data, degree)
                polynomial = np.polyval(coeffs, x)
        
        else:
            raise ValueError(f"Unknown fit_method: {fit_method}. Use 'least_squares', 'absolute_deviation', or 'ransac'.")
        
        # Subtract the polynomial from the row
        img[i, :] -= polynomial
    
    return img

def threePointFlattening(image, p1, p2, p3, mask=None):
    """
    Perform three-point plane leveling on an AFM topography image.
    
    Args:
        image (np.ndarray): 2D array of topography data
        p1, p2, p3: Three points to define the plane
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
            If provided, plane is fit to all good points instead of just 3 points.
        
    Returns:
        np.ndarray: Flattened image with plane defined by three corners subtracted
    """
    img = image.copy()
    rows, cols = img.shape
    
    # If mask is provided, use planeLevelFlattening instead (which supports masks)
    if mask is not None:
        return planeLevelFlattening(img, mask=mask)
    
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
    
    # Subtract the plane from the image
    return img - plane

def ZeroImage(image, mask=None, bins=100, min_peak_height=0.02, smoothing=5):
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
        # Create a copy of the input image
        img = image.copy()
        
        # Apply mask if provided
        if mask is not None:
            valid_data = img[mask == 1]
            if len(valid_data) == 0:  # If mask excludes all points, use all data
                valid_data = img.flatten()
        else:
            valid_data = img.flatten()
        
        # Create histogram
        hist, bin_edges = np.histogram(valid_data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Apply smoothing to the histogram to reduce noise
        if smoothing > 0:
            hist_smooth = ndimage.gaussian_filter1d(hist, sigma=smoothing)
        else:
            hist_smooth = hist
        
        # Normalize the histogram
        hist_norm = hist_smooth / np.max(hist_smooth)
        
        # Find peaks (local maxima) in the histogram
        peaks = []
        for i in range(1, len(hist_norm)-1):
            if hist_norm[i] > hist_norm[i-1] and hist_norm[i] > hist_norm[i+1] and hist_norm[i] >= min_peak_height:
                peaks.append((bin_centers[i], hist_norm[i]))
        
        # If no significant peaks found, just return original image
        if not peaks:
            return img
        
        # Sort peaks by height (frequency) in descending order
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Take the most common height as the floor/zero level
        floor_level = peaks[0][0]
        
        # Adjust the image so the floor is at zero
        zeroed_image = img - floor_level
        
        return zeroed_image

def facetLevelTiltFlattening(image, mask=None):
    """
    Perform facet level tilt flattening on an AFM topography image.
    
    Args:
        image (np.ndarray): 2D array of topography data
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        
    Returns:
        np.ndarray: Flattened image with optimal facet-preserving plane subtracted
    """
    img = image.copy()
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
    
    # Subtract the plane from the image
    return img - plane

def terraceLineFlattening(image, threshold=None, mask=None):
    """
    Perform terrace flattening on an AFM topography image by identifying horizontal terraces.
    
    Args:
        image (np.ndarray): 2D array of topography data
        threshold (float, optional): Threshold for determining terraces, if None it's calculated automatically
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        
    Returns:
        np.ndarray: Flattened image with terrace model subtracted
    """
    
    # Create a copy of the input image
    img = image.copy()
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
    
    # Create a grid of sample points
    x_grid = np.linspace(0, cols-1, min(20, cols))
    y_grid = np.linspace(0, rows-1, min(20, rows))
    X, Y = np.meshgrid(x_grid, y_grid)
    
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
    return img - terrace_img

def SetPrefix(y):
    y=np.array(y)
    # Determine best unit prefix for current values
    y_abs_max = max(abs(y.min()), abs(y.max()))
    if y_abs_max < 1e-12:
        scale = 1e15
        prefix = 'f'
    if y_abs_max < 1e-9:
        scale = 1e12
        prefix = 'p'
    elif y_abs_max < 1e-6:
        scale = 1e9
        prefix = 'n'
    elif y_abs_max < 1e-3:
        scale = 1e6
        prefix = 'μ'
    elif y_abs_max < 1:
        scale = 1e3
        prefix = 'm'
    else:
        scale = 1
        prefix = ''    
    return scale, prefix

def AFMPlot(images, show_histogram=False, **kwargs):
    
    if isinstance(images, dict)== False:
        raise ValueError("images should be a dictionary loaded with MI.loadMI, or a single image from the dictionary")
        
    showPlot=True
    #if images is a dictionary, extract the image data
    if 'images' in images.keys():
        isMultipleChannels = True
        keys = list(images['images'].keys())
        cols = 2
        rows = int(np.ceil(len(keys)/cols))
        
        fig, ax = plt.subplots(rows, cols, **kwargs)
    else:
        isMultipleChannels = False
        #check if kwargs already has a figure and axis
        if 'fig' in kwargs.keys() and 'ax' in kwargs.keys():
            fig = kwargs['fig']
            ax = kwargs['ax']
            showPlot=False
        else:
            # Create a new figure and axes
            # Filter out 'title' and 'mask' from kwargs for plt.subplots
            subplot_kwargs = {k: v for k, v in kwargs.items() if k not in ['title', 'mask','segments','show_region_labels']}
            
            # If show_histogram is True and this is a single channel image without assigned fig/ax,
            # create a figure with two subplots side by side
            if show_histogram and not isMultipleChannels:
                # Adjust figure size if provided
                if 'figsize' in subplot_kwargs:
                    orig_width, height = subplot_kwargs['figsize']
                    subplot_kwargs['figsize'] = (orig_width * 2, height)  # Double the width
                
                fig, ax = plt.subplots(1, 2, **subplot_kwargs)
            else:
                fig, ax = plt.subplots(1, 1, **subplot_kwargs)
            

    #remove ax from kwargs if it exists
    if 'ax' in kwargs.keys():
        del kwargs['ax']
            
    if isMultipleChannels:
        for i, key in enumerate(keys):
            image = images['images'][key] 
            _SinglePlot(ax[i//2, i%2], image, **kwargs)
    else:
        if show_histogram and not isinstance(ax, plt.Axes):
            # Plot the image on the first subplot
            _SinglePlot(ax[0], images, **kwargs)
            
            # Extract image data for histogram
            if isinstance(images, dict) and 'img' in images:
                img_data = images['img'].ravel()
                unit = images.get('unit', '')
                
                # Get same scaling as used in _SinglePlot
                if unit == 'um':
                    img_data = img_data * 1e-6
                    unit = 'm'
                
                scale, prefix = SetPrefix(img_data)
                img_data = img_data * scale
                
                # Determine normalization range for consistent coloring with image
                if 'vrange' in kwargs:
                    vmin, vmax = np.percentile(img_data, kwargs['vrange'])
                else:
                    vmin, vmax = np.percentile(img_data, [1, 99])
                
                # Get the same colormap as used in _SinglePlot
                cmap = plt.cm.get_cmap('afmhot')
                
                # Create histogram on the second subplot
                ax[1].set_title('Height Distribution', fontsize=12, fontweight='bold')

                # Calculate histogram
                hist, bin_edges = np.histogram(img_data, bins=100, range=(vmin, vmax))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # Normalize bin heights for visualization
                hist_norm = hist / np.max(hist)

                # Set figure size to match or be slightly smaller than the image subplot
                fig = ax[0].figure
                bbox = ax[0].get_position()
                ax[1].set_position([bbox.x0 + bbox.width * 1.05, bbox.y0, bbox.width * 0.9, bbox.height])

                # Draw bars with colors from the colormap
                for i, (x, h) in enumerate(zip(bin_centers, hist)):
                    # Normalize x to [0, 1] range for colormap
                    color_val = (x - vmin) / (vmax - vmin) if vmax > vmin else 0
                    color_val = max(0, min(1, color_val))  # Clamp to [0, 1]
                    ax[1].bar(x, hist_norm[i], width=(bin_edges[i+1] - bin_edges[i]), 
                            color=cmap(color_val), edgecolor=cmap(color_val), alpha=0.8)

                ax[1].set_xlabel(f'Height ({prefix}{unit})', fontsize=12, fontweight='bold')
                ax[1].set_ylabel('Normalized Frequency', fontsize=12, fontweight='bold')
                ax[1].grid(True, alpha=0.3)

                
        else:
            # Regular single plot without histogram
            _SinglePlot(ax, images, **kwargs)
        
    if showPlot:
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.show()
            
class AFMFlatteningMethod(Enum):
    Median = 1 
    MedianOfDifference = 2
    Modus = 3 
    Matching = 4 
    FacetLevelTilt = 5 
    PolynomialLine = 6
    TrimmedMean = 7
    TrimmedMeanOfDifference = 8 
    TerraceLine = 9        
    PlaneLevel=10 
    ThreePoint=11
    polynomialPlaneFlattening= 12
    TerracePlanes= 13
    
class AFMMaskGenerator(Enum):
    HighPoints = 1  # Remove regions that are too high
    StreakMask = 13  # Detect scan-direction streaks caused by persistent tip damage or contamination
    TrimmedMean = 3  # Mask the points that are well out of the normal from the image
    ParticleDetection = 4  # Find particles in the image and mask them
    Unsmoothable = 5  # Smooth image and remove the points that do not smooth well
    EdgeArtifacts = 7  # Remove or downweight edges where tip lift-off or overshoot occurs
    SpikeArtifacts = 10  # Mask single-pixel or very small clusters of anomalous height (z-spikes)
    LowPoints = 11  # Analogous to HighPoints but for unusually deep wells (e.g., pits or holes)
    SharpEdges = 12  # Mask sharp edges or features that are not smooth
    
class EdgeDetectionMethod(Enum):
    Sobel = 1
    Prewitt = 2
    Roberts = 3    
    Canny =4 
    
def AutoMask(image, method=AFMMaskGenerator.Unsmoothable, **kwargs):    
    """
    Generate a mask for AFM image flattening using the specified method.
    Args:
        image (np.ndarray): 2D array of topography data
        method (AFMMaskGenerator): Method to use for generating the mask
        **kwargs: Additional parameters specific to the mask generation method
    Returns:
        np.ndarray: Boolean mask where 1 indicates good points to use for flattening
    """
    
    if 'img' in image:
        image = image['img']
        
    if method == AFMMaskGenerator.HighPoints:
        mask = highPointsMask(image, **kwargs)
    elif method == AFMMaskGenerator.StreakMask:
        mask = streakMask(image, **kwargs)
    elif method == AFMMaskGenerator.TrimmedMean:
        mask = trimmedMeanMask(image, **kwargs)
    elif method == AFMMaskGenerator.ParticleDetection:
        mask = particleDetectionMask(image, **kwargs)
    elif method == AFMMaskGenerator.Unsmoothable:
        mask = unsmoothableMask(image, **kwargs)
    elif method == AFMMaskGenerator.EdgeArtifacts:
        mask = edgeArtifactsMask(image, **kwargs)
    elif method == AFMMaskGenerator.SpikeArtifacts:
        mask = spikeArtifactsMask(image, **kwargs)
    elif method == AFMMaskGenerator.LowPoints:
        mask = lowPointsMask(image, **kwargs)
    elif method == AFMMaskGenerator.SharpEdges:
        mask = sharpEdgesMask(image, **kwargs)
    
    return mask

def sharpEdgesMask(image, threshold_factor=1.0, filter_method=EdgeDetectionMethod.Sobel, **kwargs):
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

def highPointsMask(image, threshold_percentile=90, **kwargs):
    """
    Generate a mask that excludes high points above a percentile threshold.
    
    Args:
        image (np.ndarray): 2D array of topography data
        threshold_percentile (float): Percentile threshold above which points are masked
        **kwargs: Additional parameters (unused)
    
    Returns:
        np.ndarray: Boolean mask where 1 indicates good points to use for flattening
    """
    # Calculate the threshold height based on percentile
    threshold = np.percentile(image, threshold_percentile)
    
    # Create mask where points below threshold are marked as good (1)
    mask = np.where(image < threshold, 1, 0)
    
    return mask

def lowPointsMask(image, threshold_percentile=10, **kwargs):
    """
    Generate a mask that excludes low points below a percentile threshold.
    
    Args:
        image (np.ndarray): 2D array of topography data
        threshold_percentile (float): Percentile threshold below which points are masked
        **kwargs: Additional parameters (unused)
    
    Returns:
        np.ndarray: Boolean mask where 1 indicates good points to use for flattening
    """
    # Calculate the threshold height based on percentile
    threshold = np.percentile(image, threshold_percentile)
    
    # Create mask where points above threshold are marked as good (1)
    mask = np.where(image > threshold, 1, 0)
    
    return mask

def streakMask(image, scan_direction='horizontal', std_threshold=2.0, min_length=10, **kwargs):
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

def trimmedMeanMask(image, trim_factor=3.0, **kwargs):
    """
    Generate a mask that excludes points that are far from the trimmed mean.
    
    Args:
        image (np.ndarray): 2D array of topography data
        trim_factor (float): Factor multiplied by MAD to determine outlier threshold
        **kwargs: Additional parameters (unused)
    
    Returns:
        np.ndarray: Boolean mask where 1 indicates good points to use for flattening
    """
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

def particleDetectionMask(image, threshold_method='otsu', min_size=10, **kwargs):
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

def unsmoothableMask(image, filter_size=5, threshold_factor=2.0, **kwargs):
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

def edgeArtifactsMask(image, edge_width=5, method=EdgeDetectionMethod.Sobel, threshold=0.5, **kwargs):
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

def spikeArtifactsMask(image, threshold_sigma=5.0, neighborhood_size=3, **kwargs):
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
    
def terracePlaneFlattening(image, mask=None, sectionPercent=4, polyDegree=3, limitPercent=0.75):
    """
    A method to flatten the AFM image by dividing it into many sections and then using the normal of each 
    section to fit a polynomial plane to the image. This method is useful for images with terraces or steps.
    
    Args:
        image (np.ndarray): 2D array of topography data
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for flattening.
        sectionPercent (float): The size of the sections to divide the image into, as a percentage of the image size.
        polyDegree (int): Degree of 2D polynomial to fit (max is 3)
        limitPercent (float): Percentage of the image to use for fitting the polynomial plane, used to avoid edges and spots.
        
    Returns:
        np.ndarray: Flattened image with the polynomial plane subtracted without the offset from terraces.
    """
    rows, cols = image.shape
    sections = int(np.ceil(sectionPercent / 100 * rows))
    
    # Create coordinate meshgrid
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Copy the input image
    img = np.copy(image)
    
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
            section = img[y_start:y_end, x_start:x_end]
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
        return img
    
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
    return image - plane
    
def FlattenImage(imagePack, flattenMethod=AFMFlatteningMethod.PlaneLevel, mask=None, **kwargs):
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
    
    if flattenMethod == AFMFlatteningMethod.PlaneLevel:
        topography = planeLevelFlattening(imagePack['img'], **kwargs)
    elif flattenMethod == AFMFlatteningMethod.Median:
        topography = medianFlattening(imagePack['img'], **kwargs)
    elif flattenMethod == AFMFlatteningMethod.TrimmedMean:
        topography = _trimmed_mean_flattening(imagePack['img'], **kwargs)
    elif flattenMethod == AFMFlatteningMethod.TrimmedMeanOfDifference:
        topography = trimmed_mean_of_difference_flattening(imagePack['img'], **kwargs)
    elif flattenMethod == AFMFlatteningMethod.MedianOfDifference:
        topography = medianOfDifferenceFlattening(imagePack['img'], **kwargs)
    elif flattenMethod == AFMFlatteningMethod.Modus:
        topography = modusFlattening(imagePack['img'], **kwargs)
    elif flattenMethod == AFMFlatteningMethod.Matching:
        topography = matchingFlattening(imagePack['img'], **kwargs)
    elif flattenMethod == AFMFlatteningMethod.PolynomialLine:
        topography = polynomialLineFlattening(imagePack['img'], **kwargs)
    elif flattenMethod == AFMFlatteningMethod.TerraceLine:
        topography = terraceLineFlattening(imagePack['img'], **kwargs)
    elif flattenMethod == AFMFlatteningMethod.FacetLevelTilt:
        topography = facetLevelTiltFlattening(imagePack['img'], **kwargs)
    elif flattenMethod == AFMFlatteningMethod.ThreePoint:
        topography = threePointFlattening(imagePack['img'], **kwargs)
    elif flattenMethod == AFMFlatteningMethod.polynomialPlaneFlattening:
        topography = polynomialPlaneFlattening(imagePack['img'], **kwargs)
    elif flattenMethod == AFMFlatteningMethod.TerracePlanes:
        topography = terracePlaneFlattening(imagePack['img'], **kwargs)

    # Copy imagepack to a new variable, then add the flattened image
    imagePack = imagePack.copy()
    imagePack['img'] = topography    
    return imagePack

class EdgeDetectionMethod(Enum):
    Sobel = 1
    Prewitt = 2
    Roberts = 3    
    Canny = 4 
    
    import matplotlib.pyplot as plt
        
def _SinglePlot(ax, imagePack, **kwargs):
    
    # Define width and height in micrometers
    if imagePack['widthUnit'] == 'um':
        iWidth = imagePack['width'] * 1e-6
    else:
        iWidth = imagePack['width']  
    if imagePack['heightUnit'] == 'um':
        iHeight = imagePack['height'] * 1e-6
    else:
        iHeight = imagePack['height']
    
    scaleXY, prefixXY = SetPrefix([iWidth])
    
    dataUnit = imagePack['unit']
    if dataUnit == 'um':
        dataUnit = 'm'
        image = imagePack['img'] * 1e-6
    else:
        image = imagePack['img']
    
    scaleZ, prefixZ = SetPrefix(image.ravel())
    
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'{imagePack["label"]} - {imagePack["direction"]}', fontsize=12, fontweight='bold')
        
    # Determine normalization range, excluding outliers
    if 'vrange' in kwargs:
        vmin, vmax = np.percentile(image, kwargs['vrange'])
    else:
        vmin, vmax = np.percentile(image, [1, 99])  # 1st and 99th percentiles
        
    # Display the AFM data
    im = ax.imshow(
        image * scaleZ,
        cmap='afmhot',
        extent=[0, iWidth * scaleXY, 0, iHeight * scaleXY],
        interpolation='nearest',
        norm=Normalize(vmin=vmin * scaleZ, vmax=vmax * scaleZ),  # Normalize with determined range
        origin='lower'  # Ensure the origin is at the bottom left to correctly align with segments
    )
    
    # Add a transparent red overlay for masked regions if mask is in kwargs
    if 'mask' in kwargs:
        mask = kwargs['mask']
        # Create a red overlay for masked areas
        mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
        mask_overlay[mask == 0, 0] = 1  # Red channel
        mask_overlay[mask == 0, 3] = 0.5  # Alpha channel (transparency)
        
        # Add the overlay to the image
        ax.imshow(mask_overlay, extent=[0, iWidth * scaleXY, 0, iHeight * scaleXY], 
                    interpolation='nearest', origin='lower')
    
    # Add colored segment outlines if 'segments' is provided
    if 'segments' in kwargs:
        segments = kwargs['segments']
        
        # If segments is a dictionary (from segment_afm_image function)
        if isinstance(segments, dict) and 'labeled_image' in segments:
            labeled_image = segments['labeled_image']
            num_regions = labeled_image.max()
            
            # Get colors from a colormap
            cmap = plt.cm.tab20
            colors = [cmap(i % cmap.N) for i in range(num_regions)]
            
            # Draw boundaries between regions
            boundaries = segmentation.find_boundaries(labeled_image, mode='thick')
            rows, cols = boundaries.shape
            y_scale = iHeight * scaleXY / rows
            x_scale = iWidth * scaleXY / cols
            
            # Plot region boundaries with unique colors
            for i in range(1, num_regions + 1):
                # Create perimeter mask for this region
                region_mask = labeled_image == i
                perimeter = segmentation.find_boundaries(region_mask, mode='outer')
                
                # Get perimeter coordinates
                y_coords, x_coords = np.where(perimeter)
                
                # Scale coordinates to image extent
                x_scaled = x_coords * x_scale
                y_scaled = y_coords * y_scale
                
                # Plot perimeter with unique color
                if len(x_scaled) > 0:
                    ax.scatter(x_scaled, y_scaled, color=colors[i-1], s=1, alpha=0.8)
                    
                    # Optionally add region labels
                    if kwargs.get('show_region_labels', False):
                        # Find center of region
                        props = measure.regionprops(region_mask.astype(int))[0]
                        y_center, x_center = props.centroid
                        x_center_scaled = x_center * x_scale
                        y_center_scaled = y_center * y_scale
                        
                        # Add label
                        ax.text(x_center_scaled, y_center_scaled, f"{i}", 
                                color='white', fontsize=10, ha='center', va='center',
                                bbox=dict(facecolor=colors[i-1], alpha=0.7, boxstyle='round'))
        
        # If segments is a list of binary masks
        elif isinstance(segments, list) and all(isinstance(seg, np.ndarray) for seg in segments):
            # Get colors from a colormap
            cmap = plt.cm.tab20
            colors = [cmap(i % cmap.N) for i in range(len(segments))]
            
            rows, cols = segments[0].shape
            y_scale = iHeight * scaleXY / rows
            x_scale = iWidth * scaleXY / cols
            
            # Plot each segment
            for i, segment in enumerate(segments):
                # Get perimeter of segment
                perimeter = segmentation.find_boundaries(segment, mode='outer')
                
                # Get perimeter coordinates
                y_coords, x_coords = np.where(perimeter)
                
                # Scale coordinates to image extent
                x_scaled = x_coords * x_scale
                y_scaled = y_coords * y_scale
                
                # Plot perimeter with unique color
                if len(x_scaled) > 0:
                    ax.scatter(x_scaled, y_scaled, color=colors[i], s=1, alpha=0.8)
    
    # Set ticks and labels
    ax.set_xticks([0, iWidth * scaleXY])
    ax.set_yticks([0, iHeight * scaleXY])
    ax.set_xticklabels(['0', f'{iWidth * scaleXY:.1f}'], fontsize=12, fontweight='bold')
    ax.set_yticklabels(['0', f'{iHeight * scaleXY:.1f}'], fontsize=12, fontweight='bold', rotation='vertical')

    # Set axis labels
    ax.set_xlabel(f'Width ({prefixXY}m)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Height ({prefixXY}m)', fontsize=12, fontweight='bold')

    # Add the color bar with the same height as the image
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel(f'({prefixZ}{dataUnit})', fontsize=12, fontweight='bold')

    # Remove grid
    ax.grid(False)


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
    
    # Get nice scales for display
    scale_xy, prefix_xy = SetPrefix([width_m, height_m])
    scale_z, prefix_z = SetPrefix(img_data_m.ravel())
    
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

def ZeroImageToFloor(image, mask=None):
    """
    Sets the image zero level to the most likely floor level by analyzing histogram peaks.
    
    Args:
        image (np.ndarray or dict): 2D array of topography data or image dictionary
        mask (np.ndarray, optional): Boolean mask where 1 indicates good points to use for analysis
    Returns:
        np.ndarray or dict: Zeroed image with floor level at zero
    """
    bins=100
    peak_prominence=0.05
    smoothing=3
    # Extract image if dictionary provided
    if isinstance(image, dict) and 'img' in image:
        img_array = image['img'].copy()
        return_dict = True
    else:
        img_array = image.copy()
        return_dict = False
    
    # Apply mask if provided
    if mask is not None:
        valid_data = img_array[mask == 1]
        if len(valid_data) == 0:  # If mask excludes all points, use all data
            valid_data = img_array.flatten()
    else:
        valid_data = img_array.flatten()
    
    # Create histogram
    hist, bin_edges = np.histogram(valid_data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Apply smoothing to the histogram
    if smoothing > 0:
        hist_smooth = ndimage.gaussian_filter1d(hist, sigma=smoothing)
    else:
        hist_smooth = hist
    
    # Normalize the histogram
    hist_norm = hist_smooth / np.max(hist_smooth)
    
    # Find peaks in the histogram
    peaks = []
    for i in range(1, len(hist_norm)-1):
        # Check if this point is higher than its neighbors
        if hist_norm[i] > hist_norm[i-1] and hist_norm[i] > hist_norm[i+1]:
            # Calculate prominence (height above higher neighbor)
            left_min = np.min(hist_norm[:i]) if i > 0 else hist_norm[i]
            right_min = np.min(hist_norm[i+1:]) if i < len(hist_norm)-1 else hist_norm[i]
            prominence = hist_norm[i] - max(left_min, right_min)
            
            # Only include peaks with sufficient prominence
            if prominence >= peak_prominence:
                peaks.append((bin_centers[i], hist_norm[i], prominence))
    
    # If no significant peaks found, return original image
    if not peaks:
        if return_dict:
            return image
        else:
            return img_array
    
    # Sort peaks by height (frequency) in descending order
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    # Find the lowest peak with significant height (likely the floor/substrate)
    significant_peaks = [p for p in peaks if p[1] >= 0.2]  # Only consider peaks with at least 20% of max height
    
    if significant_peaks:
        # Among significant peaks, choose the one with lowest height value
        floor_level = min(significant_peaks, key=lambda x: x[0])[0]
    else:
        # If no significant peaks, just use the highest peak
        floor_level = peaks[0][0]
    
    # Adjust the image so the floor is at zero
    zeroed_image = img_array - floor_level
    
    # Return in the same format as input
    if return_dict:
        image_copy = image.copy()
        image_copy['img'] = zeroed_image
        return image_copy
    else:
        return zeroed_image
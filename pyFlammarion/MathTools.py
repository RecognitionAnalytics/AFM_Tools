
from numpy.polynomial.polynomial import Polynomial

def fast_polynomial_background(shape, coeffs):
    """Optimized polynomial background calculation."""
    ny, nx = shape
    # Create normalized coordinates once
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y)
    
    # Directly compute 3rd order polynomial using array operations
    z = (coeffs[0] + 
         coeffs[1]*xx + coeffs[2]*yy +
         coeffs[3]*xx**2 + coeffs[4]*xx*yy + coeffs[5]*yy**2 +
         coeffs[6]*xx**3 + coeffs[7]*xx**2*yy + coeffs[8]*xx*yy**2 + coeffs[9]*yy**3)
    return z



def fast_peak_quality(heights):
    """
    Evaluate quality of flattening based on the main peak's standard deviation.
    Uses numpy's histogram for speed.
    """
    hist, bin_edges = np.histogram(heights, bins=50)
    peak_idx = np.argmax(hist)
    
    # Get range around main peak
    peak_center = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
    mask = np.abs(heights - peak_center) < (bin_edges[1] - bin_edges[0]) * 2
    
    if np.sum(mask) < 10:  # Avoid tiny peaks
        return 1e6
    
    # Calculate std dev of points near peak
    std_dev = np.std(heights[mask])
    return std_dev

def fast_loss_function(coeffs, topography):
    """Simplified loss function using only the main peak's standard deviation."""
    background = fast_polynomial_background(topography.shape, coeffs)
    flattened = topography - background
    return fast_peak_quality(flattened.ravel())



def fast_peak_quality(heights):
    """
    Evaluate quality of flattening based on the main peak's standard deviation.
    Uses numpy's histogram for speed.
    """
    hist, bin_edges = np.histogram(heights, bins=50)
    peak_idx = np.argmax(hist)
    
    # Get range around main peak
    peak_center = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
    mask = np.abs(heights - peak_center) < (bin_edges[1] - bin_edges[0]) * 2
    
    if np.sum(mask) < 10:  # Avoid tiny peaks
        return 1e6
    
    # Calculate std dev of points near peak
    m= np.sum(hist)
    std_dev = np.std(heights[mask])
    return  (m-np.max(hist))/m, std_dev /(bin_edges[-1] - bin_edges[0])


def peakQ_loss_function(coeffs, topography):
    """Simplified loss function using only the main peak's standard deviation."""
    background = MI.fast_polynomial_background(topography.shape, coeffs)
    flattened = topography - background
    _,Q = fast_peak_quality(flattened.ravel())
    return Q
        
def peakH_loss_function(coeffs, topography):        
    """Simplified loss function using the main peak's standard deviation and height."""
    background = MI.fast_polynomial_background(topography.shape, coeffs)
    flattened = topography - background
    peak,Q = fast_peak_quality(flattened.ravel())
    return  peak  *Q + np.sum(np.abs(coeffs )) * 1e-8




def row_by_row_polynomial_subtraction(image, poly_order):
    """
    Perform row-by-row polynomial background subtraction from the AFM image.
    
    Parameters:
    -----------
    image : 2D numpy array
        Height data from AFM measurement
    poly_order : int
        Order of the polynomial to fit and subtract
    
    Returns:
    --------
    new_image : 2D numpy array
        Image after polynomial background subtraction
    background : 2D numpy array
        Calculated polynomial background
    """
    rows, cols = image.shape
    background = np.zeros_like(image)
    
    for i in range(rows):
        # Fit polynomial to the row
        p = Polynomial.fit(np.arange(cols), image[i, :], poly_order)
        # Evaluate the polynomial
        background[i, :] = p(np.arange(cols))
    
    # Subtract the background
    new_image = image - background
    
    return new_image, background

def poly4_background(coords, *coeffs):
    """Calculate 4th-degree polynomial background."""
    # Ensure coords is a tuple of X and Y
     
    x, y = coords
    terms = [
        1, x, y, x**2, x*y, y**2,
        x**3, (x**2)*y, x*(y**2), y**3,
        x**4, (x**3)*y, (x**2)*(y**2), x*(y**3), y**4
    ]
    return sum(c * t for c, t in zip(coeffs, terms))

# Fit the polynomial to the topography data
def fit_poly4(x, y, z):
    """Fit a 4th-degree polynomial to the data."""
    # Flatten the meshgrid and topography for fitting
    x_flat, y_flat, z_flat = x.ravel(), y.ravel(), z.ravel()
    # Initial guess for coefficients (all zeros)
    initial_guess = [0] * 15
    coeffs, _ = curve_fit(poly4_background, (x_flat, y_flat), z_flat, p0=initial_guess)
    return coeffs

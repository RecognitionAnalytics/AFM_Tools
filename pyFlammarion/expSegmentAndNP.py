
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy import ndimage as ndi

def load_image(image_path):
    """Loads the image and crops the footer if necessary."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return crop_footer(img)

def crop_footer(img, crop_fraction=0.1):
    """Crops the footer of the image (bottom crop_fraction)."""
    h, w = img.shape
    crop_height = int(h * (1 - crop_fraction))
    return img[:crop_height, :]

def preprocess_image(img):
    """Applies blurring and adaptive thresholding to detect electrodes."""
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def detect_electrodes(thresh):
    """Detects contours (electrodes) and creates an electrode mask."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    electrode_mask = np.zeros_like(thresh)
    cv2.drawContours(electrode_mask, contours, -1, (255), thickness=cv2.FILLED)
    return electrode_mask

def detect_nanoparticles(img):
    """Detects gold nanoparticles using a high-pass filter and thresholding."""
    high_pass = img - cv2.GaussianBlur(img, (21, 21), 3)
    _, nanoparticle_mask = cv2.threshold(high_pass, 50, 255, cv2.THRESH_BINARY)
    labels = measure.label(nanoparticle_mask)
    return measure.regionprops(labels)

def detect_edges(electrode_mask):
    """Detects the edges of the electrodes."""
    edges = cv2.Canny(electrode_mask, 100, 200)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    return dilated_edges

def classify_nanoparticles(nanoparticles, electrode_mask, dilated_edges):
    """Classifies nanoparticles as being on electrodes or on the oxide surface."""
    electrode_nps = []
    oxide_nps = []
    
    for prop in nanoparticles:
        y, x = prop.centroid
        x, y = int(x), int(y)
        
        if electrode_mask[y, x] > 0:
            if dilated_edges[y, x] > 0:
                electrode_nps.append((x, y, 'edge'))
            else:
                electrode_nps.append((x, y, 'on'))
        else:
            oxide_nps.append((x, y, 'oxide'))
    
    return electrode_nps, oxide_nps

def plot_results(img, electrode_nps, oxide_nps):
    """Displays the original image with marked nanoparticles and classifications."""
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for x, y, loc in electrode_nps:
        color = (0, 0, 255) if loc == 'edge' else (0, 255, 0)
        cv2.circle(output_img, (x, y), 5, color, 1)
    
    for x, y, loc in oxide_nps:
        cv2.circle(output_img, (x, y), 5, (255, 0, 0), 1)

    plt.figure(figsize=(10, 10))
    plt.title("Nanoparticles Classification")
    plt.imshow(output_img)
    plt.show()
    
def runExample():
    image_path =   image_dir+ "B1_Device.tif"  # Replace with your SEM image path 
    # Load and preprocess the image
    img = load_image(image_path)

    # Preprocess and detect electrodes
    thresh = preprocess_image(img)
    electrode_mask = detect_electrodes(thresh)

    # Detect nanoparticles
    nanoparticles = detect_nanoparticles(img)

    # Detect edges of electrodes
    dilated_edges = detect_edges(electrode_mask)

    # Classify nanoparticles
    electrode_nps, oxide_nps = classify_nanoparticles(nanoparticles, electrode_mask, dilated_edges)

    # Plot results
    plot_results(img, electrode_nps, oxide_nps)


def rate_rectangularity(contour):
    """Rates the rectangularity of a contour based on the aspect ratio and solidity."""
    
    # Get the bounding rectangle and calculate aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    
    # Aspect ratio for a perfect rectangle should be close to 1.0
    rect_aspect_score = min(aspect_ratio, 1 / aspect_ratio)  # Closer to 1 means more rectangular
    
    # Get the contour area and bounding box area
    contour_area = cv2.contourArea(contour)
    bounding_box_area = w * h
    
    # Solidity: ratio of contour area to the bounding box area (close to 1 for rectangular)
    if bounding_box_area > 0:
        solidity = contour_area / bounding_box_area
    else:
        solidity = 0
    
    # Combine the scores (weighted sum)
    rectangularity_score = 0.5 * rect_aspect_score + 0.5 * solidity
    
    return rectangularity_score

def mark_electrodes_with_red_outline(img, electrode_mask, min_contour_area=50000):
    """Marks the electrode area with a red outline on the SEM image, selecting only long contours."""
    img_with_outline = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    
    # Find contours of the electrodes
    contours, _ = cv2.findContours(electrode_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for long and coherent contours
    #long_contours = sorted(contours, key = lambda c: cv2.arcLength(c,True), reverse = True)
    long_contours = sorted(contours, key = lambda c: rate_rectangularity(c), reverse = True)
    #filter the contours based on how straight the line is 
    long_contours = [c for c in long_contours if rate_rectangularity(c) > 0.5]
    
    #select countours with area > min_contour_area
    selected_contours = [c for c in long_contours if cv2.contourArea(c) > min_contour_area]
    # Draw the filtered contours in red (BGR: (0, 0, 255)) on the image
    selectedMask = np.zeros_like(electrode_mask)
    #draw filled contours on the mask
    cv2.drawContours(selectedMask,selected_contours, -1, (255), thickness=cv2.FILLED)
    
    #add another countour that is the rest of the image 
    selected_contours.append( cv2.findContours(255-selectedMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
    
    cv2.drawContours(img_with_outline,selected_contours, -1, (0, 0, 255), thickness=2)
    return img_with_outline,selected_contours

def plot_image_with_electrodes(img_with_outline):
    """Plots the image with marked electrodes in a high-quality figure."""
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_with_outline, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for plotting
    plt.title("Electrode Area with Red Outline")
    plt.axis("off")  # Hide axis for a cleaner look
    plt.show()
    

def rate_linearity(contour, epsilon_factor=0.02):
    """Rates the linearity of a contour based on how straight its edges are."""
    
    # Approximate contour with reduced points (polygons)
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    total_deviation = 0
    for i in range(len(approx)):
        pt1 = tuple(approx[i][0])  # Convert to (x, y) tuple
        pt2 = tuple(approx[(i + 1) % len(approx)][0])  # Convert to (x, y) tuple
        
      
        
        # Find points on the contour that deviate from the line segment (pt1, pt2)
        for point in contour:
            
            deviation = cv2.pointPolygonTest(np.array([pt1, pt2]), tuple(point[0]), True)
            total_deviation += abs(deviation)
    
    # Normalize deviation by contour length to get a straightness score
    normalized_deviation = total_deviation / cv2.arcLength(contour, True)
    
    # Linearity score: 1.0 means perfect straight lines, lower values indicate more deviation
    linearity_score = max(1.0 - normalized_deviation, 0)
    
    return linearity_score

def rate_rectangularity_and_linearity(contour, epsilon_factor=0.02):
    """Rates the contour based on rectangularity and straightness of lines."""
    
    # Get the bounding rectangle and calculate aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    
    # Aspect ratio score (closer to 1 means more rectangular)
    rect_aspect_score = min(aspect_ratio, 1 / aspect_ratio)
    
    # Calculate solidity (area-based score)
    contour_area = cv2.contourArea(contour)
    bounding_box_area = w * h
    solidity = contour_area / bounding_box_area if bounding_box_area > 0 else 0
    
    # Combine the rectangularity score
    rectangularity_score = 0.5 * rect_aspect_score + 0.5 * solidity
    
    # Linearity score: how straight the contour's edges are
    linearity_score = rate_linearity(contour, epsilon_factor)
    
    # Combine rectangularity and linearity
    total_score = 0.5 * rectangularity_score + 0.5 * linearity_score
    
    return total_score

def filter_high_quality_contours(contours, min_score=0.85, epsilon_factor=0.02):
    """Filters contours based on both rectangularity and linearity scores."""
    filtered_contours = [c for c in contours if rate_rectangularity_and_linearity(c, epsilon_factor) >= min_score]
    return filtered_contours

def mark_high_quality_electrodes(img, electrode_mask, min_score=0.85):
    """Marks only high-quality rectangular and straight electrodes based on contour ratings."""
    img_with_outline = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Find contours of the electrodes
    contours, _ = cv2.findContours(electrode_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter the contours based on rectangularity and linearity score
    high_quality_contours = filter_high_quality_contours(contours, min_score)
    
    # Draw the filtered contours in red (BGR: (0, 0, 255)) on the image
    cv2.drawContours(img_with_outline, high_quality_contours, -1, (0, 0, 255), thickness=2)

    return img_with_outline    

def join_lines(lines, img_shape, gap_threshold=10):
    """
    Joins nearby lines that are pointing in the same direction into continuous edges.

    Parameters:
        lines (list): List of detected lines from Hough transform.
        img_shape (tuple): Shape of the image (height, width).
        gap_threshold (float): The maximum distance between endpoints of lines to be joined.

    Returns:
        list: List of joined lines as arrays of points.
    """
    if lines is None:
        return []

    lines = [line[0] for line in lines]  # Unpack lines

    def are_lines_close(line1, line2, gap_threshold):
        """Check if the two lines are close and aligned enough to be joined."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate distance between endpoints of line1 and startpoints of line2
        dist1 = np.hypot(x2 - x3, y2 - y3)
        dist2 = np.hypot(x1 - x4, y1 - y4)

        # Angle difference between lines
        angle1 = np.arctan2(y2 - y1, x2 - x1)
        angle2 = np.arctan2(y4 - y3, x4 - x3)
        angle_diff = np.abs(angle1 - angle2)

        # Check if distances and angles are within threshold
        return (dist1 < gap_threshold or dist2 < gap_threshold) and angle_diff < np.pi / 8

    def interpolate_to_edge(line, img_shape):
        """Interpolate a line to extend to the edges of the image."""
        x1, y1, x2, y2 = line
        h, w = img_shape

        # Line equation: y = mx + b
        if x2 - x1 != 0:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

            # Intersect with image borders
            x_left, x_right = 0, w
            y_left, y_right = int(b), int(m * w + b)
            y_top, y_bottom = 0, h
            x_top = int(-b / m) if m != 0 else x1
            x_bottom = int((h - b) / m) if m != 0 else x2

            points = [
                (x_left, y_left),
                (x_right, y_right),
                (x_top, 0),
                (x_bottom, h)
            ]

            # Keep only points inside image borders
            points = [(x, y) for x, y in points if 0 <= x < w and 0 <= y < h]
            return points[0], points[-1]
        else:
            return (x1, 0), (x2, h)  # Vertical lines

    joined_lines = []

    # Join lines based on proximity and direction
    for line in lines:
        added = False
        for joined_line in joined_lines:
            if are_lines_close(line, joined_line[-1], gap_threshold):
                joined_line.append(line[1:])
                added = True
                break
        if not added:
            joined_lines.append([line])

    # Interpolate lines that reach image edges
    for joined_line in joined_lines:
        x1, y1, x2, y2 = joined_line[0], joined_line[-1]
        if x1 == 0 or y1 == 0 or x2 == img_shape[1] or y2 == img_shape[0]:
            joined_line[0], joined_line[-1] = interpolate_to_edge(joined_line, img_shape)

    return joined_lines

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_bilateral,
    denoise_wavelet,
    estimate_sigma,
)


def load_image(image_path):
    """Loads the image and crops the footer if necessary."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return crop_footer(img)

def crop_footer(img, crop_fraction=0.07):
    """Crops the footer of the image (bottom crop_fraction)."""
    h, w = img.shape
    crop_height = int(h * (1 - crop_fraction))
    return img[:crop_height, :]


img = load_image(image_path)
print(img.shape)
img= (denoise_wavelet(img,   rescale_sigma=True)*255).astype(np.uint8)
print(img.shape)    
th22 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,249,2)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,249,2)

from scipy.ndimage import binary_fill_holes

 # Step 2: Preprocess the image - Gaussian Blur to smooth
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Step 3: Edge detection with Sobel operator
grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
grad_mag = cv2.magnitude(grad_x, grad_y)

# Normalize the magnitude and convert to 8-bit image
grad_mag = cv2.convertScaleAbs(grad_mag)

# Step 4: Binary thresholding to keep strong edges
_, edges = cv2.threshold(grad_mag, 50, 255, cv2.THRESH_BINARY)

# Step 6: Fill small holes in the detected edges
edges_filled = binary_fill_holes(edges).astype(np.uint8) * 255

# Step 7: Further thinning to reduce line thickness
#thin_edges = cv2.ximgproc.thinning(edges_filled)

# Step 8: Contour detection to find long continuous line segments
contours, _ = cv2.findContours(edges_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 9: Filter contours based on length
min_contour_length = 50  # You can adjust this value
long_contours = [cnt for cnt in contours if cv2.arcLength(cnt, False) > min_contour_length]
print(len(long_contours))
# Step 10: Create a blank image to draw the detected long lines
line_img2 = np.zeros_like(edges)

#draw filled contours
 
# Draw the long contours
cv2.drawContours(line_img2, long_contours, -1, (255), -1)

plt.imshow(line_img2, cmap='gray')

def filter_and_merge_contours(contours, min_contour_length=50, circularity_threshold=0.05):
    filtered_contours = []

    # Step 1: Filter out small spherical contours based on circularity
    for cnt in contours:
        if cv2.arcLength(cnt, False) > min_contour_length:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < circularity_threshold:  # Not circular
                filtered_contours.append(cnt)
    
    # Step 2: Merge contours that are aligned based on direction
    # merged_contours = []
    
    # while filtered_contours:
    #     cnt1 = filtered_contours.pop(0)
    #     merged = [cnt1]
        
    #     # Compare with remaining contours
    #     remaining_contours = []
    #     for cnt2 in filtered_contours:
    #         # Calculate direction of both contours by fitting lines
    #         [vx1, vy1, _, _] = cv2.fitLine(cnt1, cv2.DIST_L2, 0, 0.01, 0.01)
    #         [vx2, vy2, _, _] = cv2.fitLine(cnt2, cv2.DIST_L2, 0, 0.01, 0.01)
            
    #         # Calculate the angle between the two vectors
    #         dot_product = vx1 * vx2 + vy1 * vy2
    #         if abs(dot_product) > 0.95:  # Threshold for similar direction (can adjust)
    #             merged.append(cnt2)  # Add contour to current merged group
    #         else:
    #             remaining_contours.append(cnt2)  # Keep for future comparison

    #     # Combine all contours in the merged group
    #     merged_contours.append(np.vstack(merged))
    #     filtered_contours = remaining_contours  # Update list to process remaining contours
    
    return filtered_contours


# Step 5: Use Hough Line Transform to detect lines
lines = cv2.HoughLinesP(line_img2, .5, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=30)

# Step 6: Create a blank image to draw lines
line_img = np.zeros_like(img)
 
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2),  255, 2)

# Step 7: Combine the lines and original image to mark the lines
combined = cv2.addWeighted(img, 0.8, line_img, 1, 0)
plt.imshow(combined, cmap='gray')


def calculate_gradient(image):
    """
    Calculate the gradient magnitude using Sobel operator.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude

def group_hough_lines(hlines, img_width, img_height):
    """
    Group Hough lines into 4 groups based on their slope and location:
    - left edge: near the left side, mostly vertical lines
    - right edge: near the right side, mostly vertical lines
    - top edge: near the top, mostly horizontal lines
    - bottom edge: near the bottom, mostly horizontal lines
    """
    left_lines, right_lines, top_lines, bottom_lines = [], [], [], []

    for line in hlines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > abs(y2 - y1):  # Horizontal-ish lines
            if min(y1, y2) < img_height / 2:  # Top edge
                top_lines.append(line)
            else:  # Bottom edge
                bottom_lines.append(line)
        else:  # Vertical-ish lines
            if min(x1, x2) < img_width / 2:  # Left edge
                left_lines.append(line)
            else:  # Right edge
                right_lines.append(line)

    return left_lines, right_lines, top_lines, bottom_lines

def create_mask_from_lines(image_shape, lines):
    """
    Create a binary mask from the lines by drawing them on a blank image.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
    return mask

def refine_edge_using_gradient(mask, gradient):
    """
    For each vertical or horizontal position in the mask, find the pixel with the maximum gradient
    which corresponds to the edge. Return the array of edge points.
    """
    edge_points = []
    nonzero_y, nonzero_x = np.nonzero(mask)  # Get all non-zero points from the mask

    for x in range(np.min(nonzero_x), np.max(nonzero_x) + 1):
        # Find the y position with the max gradient for each x
        y_vals = nonzero_y[nonzero_x == x]
        if len(y_vals) > 0:
            y_grad = [gradient[y, x] for y in y_vals]
            max_y = y_vals[np.argmax(y_grad)]  # Find y with max gradient
            edge_points.append((x, max_y))

    return edge_points

def find_edges_from_hough_lines(image, hlines):
    """
    Main procedure that groups the Hough lines, creates masks, and refines edges using gradient data.
    Returns 4 arrays of points representing the left, right, top, and bottom edges.
    """
    # Step 1: Calculate the gradient magnitude of the image
    gradient_magnitude = calculate_gradient(image)

    # Step 2: Group the Hough lines into 4 edge regions
    img_height, img_width = image.shape
    left_lines, right_lines, top_lines, bottom_lines = group_hough_lines(hlines, img_width, img_height)

    # Step 3: Create masks for each edge region
    left_mask = create_mask_from_lines(image.shape, left_lines)
    right_mask = create_mask_from_lines(image.shape, right_lines)
    top_mask = create_mask_from_lines(image.shape, top_lines)
    bottom_mask = create_mask_from_lines(image.shape, bottom_lines)

    # Step 4: Refine the edges by using gradient information
    left_edge = refine_edge_using_gradient(left_mask, gradient_magnitude)
    right_edge = refine_edge_using_gradient(right_mask, gradient_magnitude)
    top_edge = refine_edge_using_gradient(top_mask, gradient_magnitude)
    bottom_edge = refine_edge_using_gradient(bottom_mask, gradient_magnitude)

    return left_edge, right_edge, top_edge, bottom_edge

import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter1d

def calculate_gradient(image):
    """
    Calculate the gradient magnitude using the Sobel operator.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude

def cluster_lines_by_intercept(hlines, eps=20, min_samples=3):
    """
    Cluster horizontal lines by their y-intercept values using DBSCAN.
    This helps to identify multiple horizontal edges in the image.
    """
    intercepts = []

    for line in hlines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > abs(y2 - y1):  # Horizontal-ish lines
            intercept = y1 - (y2 - y1) / (x2 - x1) * x1  # Calculate y-intercept
            intercepts.append([intercept])

    intercepts = np.array(intercepts)

    # Perform clustering on y-intercepts to group lines into horizontal edges
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(intercepts)
    return clustering.labels_

def create_mask_from_lines(image_shape, lines):
    """
    Create a binary mask from the lines by drawing them on a blank image.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
    return mask

def vertical_gradient_search(mask, gradient):
    """
    Perform vertical gradient search in the mask to find the strongest gradient point.
    Returns an array of refined edge points sorted by x-position.
    """
    edge_points = []
    nonzero_y, nonzero_x = np.nonzero(mask)  # Get all non-zero points from the mask

    for x in range(np.min(nonzero_x), np.max(nonzero_x) + 1):
        # Find the y position with the max gradient for each x
        y_vals = nonzero_y[nonzero_x == x]
        if len(y_vals) > 0:
            y_grad = [gradient[y, x] for y in y_vals]
            max_y = y_vals[np.argmax(y_grad)]  # Find y with max gradient
            edge_points.append((x, max_y))

    return np.array(edge_points)

def smooth_edge(edge_points, sigma=2):
    """
    Smooth the edge points using Gaussian filtering along the y-coordinate.
    This helps to remove noise and produce a smoother edge line.
    """
    x_vals = edge_points[:, 0]
    y_vals = edge_points[:, 1]
    
    # Apply Gaussian smoothing to the y-values
    y_smoothed = gaussian_filter1d(y_vals, sigma=sigma)
    
    # Recombine the x and smoothed y values
    smoothed_points = np.column_stack((x_vals, y_smoothed))
    
    return smoothed_points

def refine_edges(image, hlines):
    """
    Main procedure that clusters horizontal Hough lines, creates masks,
    and refines the edges using vertical gradient search.
    Returns arrays of points for the horizontal edges.
    """
    # Step 1: Calculate gradient magnitude of the image
    gradient_magnitude = calculate_gradient(image)

    # Step 2: Cluster horizontal lines by y-intercept
    labels = cluster_lines_by_intercept(hlines)
    unique_labels = set(labels)

    # Step 3: For each cluster, create a mask and refine the edge using gradient search
    img_shape = image.shape
    edge_groups = []
    
    for label in unique_labels:
        if label == -1:
            continue  # Ignore noise
        
        # Filter lines belonging to the current cluster
        cluster_lines = [hlines[i] for i in range(len(hlines)) if labels[i] == label]
        
        # Step 4: Create a mask for the current cluster
        mask = create_mask_from_lines(img_shape, cluster_lines)

        # Step 5: Perform vertical gradient search to refine the edge
        edge_points = vertical_gradient_search(mask, gradient_magnitude)
        
        # Step 6: Sort the points by x-position
        edge_points = edge_points[edge_points[:, 0].argsort()]
        
        # Step 7: Smooth the edge points
        smoothed_points = smooth_edge(edge_points)

        edge_groups.append(smoothed_points)

    return edge_groups

imageSummaries = {}
# Load the SEM image  
for imageFile in images:
    try:
        image_path =   image_dir+ imageFile  +'.tif' # Replace with your SEM image path 
        image, meta = load_image(image_path )

        pixelWidthnm, pixelHeightnm = meta['FEI_HELIOS']['EScan']['PixelWidth']*1e9,meta['FEI_HELIOS']['EScan']['PixelHeight']*1e9 

        filtered_image = denoise_bilateral(image, sigma_color=0.05, sigma_spatial=15 )
        filtered_image=(filtered_image*255).astype(np.uint8)

        inpainted_image=FindNPs(filtered_image)
        segmented=SegmentCleaned(inpainted_image)
        rotated=clip_maximum_rectangle_from_mask(filtered_image, (segmented==1).astype(np.uint8))
        np_mask,num_markers,markers,markerStats=FindFlattenedNPs(rotated)

        _,ax = plt.subplots(1,3,figsize=(3*3,3))
        ax[0].imshow(inpainted_image,cmap='gray')
        plt.axis('off')
        ax[0].set_title('Inpainted Image')
        ax[1].imshow(segmented,cmap='gray')
        ax[1].set_title('Segmented Image')
        plt.axis('off')
        ax[2].imshow(rotated,cmap='gray')
        ax[2].set_title('Central Plateau')
        plt.axis('off')
        plt.suptitle(imageFile)
        plt.show()

        particle_stats = calculate_particle_stats_with_contours(num_markers,markers,markerStats, rotated, pixelWidthnm, pixelHeightnm)
        neighbors= find_nearest_neighbors_angular_optimized(particle_stats, 300, angular_step=10)
        density_map=calculate_density_graph(np_mask, pixelWidthnm, particle_stats, blur_factor=0.1)    
        imageStats = SummaryStats(density_map, particle_stats, neighbors, np_mask, pixelWidthnm, pixelHeightnm)
        imageStats['densityMap']=density_map.ravel()
        
        del image,meta, filtered_image, inpainted_image, segmented, rotated, np_mask, num_markers, markers, markerStats, particle_stats, neighbors, density_map
        imageSummaries[imageFile] = imageStats
    except Exception as e:
        print(f"Error in processing {imageFile}: {e}")
        continue
    
    
    
from PIL import Image
import tifffile
import cv2
import numpy as np
from skimage import filters
from skimage.measure import label, regionprops
from skimage.restoration import denoise_bilateral
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt

def extract_tiff_metadata(tiff_file_path):
    """
    Extracts all tags, parameters, and hidden information from a TIFF file,
    specifically tailored for SEM images with legends containing acquisition parameters.
    
    Args:
        tiff_file_path (str): Path to the TIFF image file.
    
    Returns:
        metadata (dict): Dictionary containing all recovered tags and parameters.
    """
    metadata = {}

    # Step 1: Open the TIFF file with Pillow to get basic metadata
    with Image.open(tiff_file_path) as img:
        # Basic TIFF metadata (Image Description, DPI, Resolution Unit, etc.)
        tiff_metadata = img.tag_v2
        
        for tag, value in tiff_metadata.items():
            metadata[f'TIFF Tag {tag}'] = value
    
    # Step 2: Use tifffile to extract all TIFF-specific metadata and hidden information
    with tifffile.TiffFile(tiff_file_path) as tiff:
        for page in tiff.pages:
            metadata['Description'] = page.description
            
            # Extracting all tags from the page
            for tag in page.tags.values():
                name, value = tag.name, tag.value
                metadata[name] = value

    return metadata


def load_image(image_path):
    """Loads the image and crops the footer if necessary."""
    meta=extract_tiff_metadata(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return crop_footer(img),meta

def crop_footer(img, crop_fraction=0.07):
    """Crops the footer of the image (bottom crop_fraction)."""
    h, w = img.shape
    crop_height = int(h * (1 - crop_fraction))
    return img[:crop_height, :]


def FindNPs(filtered_image):

    # Step 2: Detect AuNPs based on intensity differences
    # Apply a slight Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(filtered_image, (3, 3), 0)

    # Use the difference between original and blurred image to highlight NPs
    intensity_diff = cv2.absdiff(filtered_image, blurred_image)

    # Threshold the difference to get potential NP regions
    _, binary_mask = cv2.threshold(intensity_diff, 20, 255, cv2.THRESH_BINARY)

    # Step 3: Filter out regions that are too large or too small (1-5 pixel wide particles)
    # Label connected regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    maxP=25
    # Create a mask to keep only particles within size bounds (1-5 pixels)
    np_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):  # Skip background (label 0)
        width, height = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if 1 <= width <= maxP and 1 <= height <= maxP:
            np_mask[labels == i] = 255
            
    # Step 4: Pad the spots in the mask to enlarge the regions for better inpainting
    kernel = np.ones((5, 5), np.uint8)  # 3x3 kernel to pad spots
    padded_mask = cv2.dilate(np_mask, kernel, iterations=1)
            
    # Step 4: Remove the detected NPs using inpainting
    inpainted_image = cv2.inpaint(filtered_image, padded_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)        
    return inpainted_image

def rectangularize(filled_mask):
    mask = filled_mask.astype(np.uint8)
    min_whisker_length=1000
    for j in range(2):
        # Step 1: Create a small structuring element to isolate thin lines
        # The size of the structuring element should depend on what we consider 'thin'
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (13, 13))

        # Step 2: Perform morphological opening to remove thin connections (whiskers)
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        removed_whiskers = mask - opened_mask
        num_labels, whisker_labels = cv2.connectedComponents(removed_whiskers)
       

        for i in range(1, num_labels):  # Start from 1 to ignore background label 0
            whisker_region = (whisker_labels == i)
            if np.sum(whisker_region) < min_whisker_length:
                # If the whisker region is smaller than the threshold, remove it
                mask[whisker_region] =0
        
        
        num_mask_labels, mask_labels = cv2.connectedComponents(mask)
        small_region_mask = np.zeros_like(mask)
        regionSizes = [np.sum(mask_labels == i) for i in range(0, num_mask_labels)]
        regionSizes[0]=0
        biggest_region = np.argmax(regionSizes)
        
        region = (mask_labels == biggest_region)
        small_region_mask[region] = 255    
        mask = small_region_mask
        
    return mask

def SegmentCleaned(inpainted_image):
    thresholds = filters.threshold_multiotsu(inpainted_image, classes=4)
    regions = np.digitize(inpainted_image, bins=thresholds)
        
    labeled_image, _ = label(regions, return_num=True, connectivity=2)
 
    # Step 3: Extract regions and sort by size (largest extent first)
    regions_sorted = regionprops(labeled_image)
    regions_sorted = sorted(regions_sorted, key=lambda r: r.area, reverse=True)

    # Step 4: Analyze regions and correct small ones
    labeledMask=np.zeros_like(labeled_image)
    regionLabel=1
    for region in regions_sorted[0:6]:
        # Get the coordinates of the small region
        coords = region.coords
        
        # Create a binary mask for the small region
        small_region_mask = np.zeros_like(regions, dtype=bool)
        small_region_mask[coords[:, 0], coords[:, 1]] = True
        
        # Fill holes in the small region (e.g., areas that are fully enclosed by boundaries)
        filled_mask = binary_fill_holes(small_region_mask)
        rect_mask = rectangularize(filled_mask)
        # Label the region in the segmented image
        labeledMask[rect_mask==255]=regionLabel
        regionLabel+=1

    return labeledMask.astype(np.uint8)

def largest_inscribed_rectangle(image, mask):
    """
    Finds the largest axis-aligned rectangle that can be inscribed within the masked region (inside the contour).
    
    Args:
        mask: Binary mask (2D numpy array) where 1s represent the area of interest and 0s represent the background.

    Returns:
        x, y, w, h: Coordinates and size (width and height) of the largest rectangle inscribed in the mask.
    """
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, max_dist, _, max_loc = cv2.minMaxLoc(dist_transform)
    max_radius = int(max_dist)
    x, y = max_loc
    top_left_x = max(x - max_radius, 0)
    top_left_y = max(y - max_radius, 0)
    width = height = 2 * max_radius

    middle_x = top_left_x + width // 2
    middle_y = top_left_y + height // 2
    #reduce the size of the rectangle by 10%
    reduce =  .1
    x1,y1,x2,y2=  int(middle_x-(1-reduce)*width//2), int(middle_y-(1-reduce)*height//2),   int(middle_x+(1-reduce)*width//2), int(middle_y+(1-reduce)*height//2)
    
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image

 
    
def clip_maximum_rectangle_from_mask(image, mask):
    """
    Extracts the maximum rectangular region from within the masked area of the image, corrects its rotation,
    and removes the background (i.e., areas outside the mask).

    Args:
        image: The original image (2D or 3D numpy array).
        mask: Binary mask (2D numpy array) where 1s represent the area of interest, and 0s represent the background.

    Returns:
        result: The cropped and rotated rectangular image from within the mask, with the background removed.
    """
    # Step 1: Find contours of the masked region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 2: Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Step 3: Find the minimum area rotated rectangle that encloses the largest contour
    rect = cv2.minAreaRect(largest_contour)

    # Step 4: Get the box points of the rectangle and order them
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Step 5: Extract the width and height of the rotated rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    # Step 6: Get the transformation matrix for rotating the image to align with the rectangle
    src_pts = box.astype(np.float32)
    dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Step 7: Warp the original image to obtain a straightened rectangle
    rotated_image = cv2.warpPerspective(image, M, (width, height))
    # Step 8: Warp the mask similarly to crop out the background from the rotated image
    rotated_mask = cv2.warpPerspective(mask, M, (width, height))
    return largest_inscribed_rectangle(rotated_image, rotated_mask)

def FindFlattenedNPs(rotated):
    thresholds = filters.threshold_multiotsu(rotated, classes=3)
    regions = np.digitize(rotated, bins=thresholds).astype(np.uint8)


    # Step 3: Filter out regions that are too large or too small (1-5 pixel wide particles)
    # Label connected regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(regions, connectivity=8)
    #maxP=25
    # Create a mask to keep only particles within size bounds (1-5 pixels)
    np_mask = np.zeros_like(rotated)
    for i in range(1,num_labels ):  # Skip background (label 0)
        width, height = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        #if 1 <= width <= maxP and 1 <= height <= maxP:
        np_mask[labels == i] = 255
        
    return np_mask,num_labels,labels,stats


def calculate_particle_stats_with_contours(num_labels, labels, stats, rotated, pixelWidthnm, pixelHeightnm):
    """
    Creates a dictionary for each detected region containing centroid, radius, area, 
    circularity difference, max intensity, average intensity, and contour-based perimeter.

    Args:
        num_labels: Number of connected components (particles).
        labels: Label matrix from connectedComponentsWithStats.
        stats: Statistics matrix from connectedComponentsWithStats.
        rotated: Rotated image for intensity analysis.

    Returns:
        particle_stats: A dictionary with particle stats for each label.
    """
    pixelArea = pixelWidthnm*pixelHeightnm
    particle_stats = {}

    for i in range(1, num_labels):  # Start from 1 to ignore background (label 0)
        # Get the stats for the current particle
        left, top, width, height, area = stats[i]

        # Step 1: Calculate the centroid
        x_center = left + width / 2
        y_center = top + height / 2
        centroid = (x_center, y_center)

        # Step 2: Create a binary mask for the current particle
        particle_mask = (labels == i).astype(np.uint8)

        # Step 3: Find contours for the current particle
        contours, _ = cv2.findContours(particle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours and len(contours) > 0:
            # Step 4: Calculate the perimeter (contour length)
            perimeter = cv2.arcLength(contours[0], True)

            # Step 5: Calculate the minimum enclosing circle radius
            (x_circle, y_circle), radius = cv2.minEnclosingCircle(contours[0])

            # Step 6: Calculate circularity: 4 * pi * Area / (Perimeter ^ 2)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0  # If no perimeter is detected
            difference_from_circular = abs(1 - circularity)  # Ideal circularity is 1
        else:
            # If no contour is found, set perimeter and circularity to 0
            perimeter = 0
            radius = 0
            difference_from_circular = 0

        # Step 7: Calculate max and average intensity from the rotated image
        particle_pixels = rotated[labels == i]
        max_intensity = np.max(particle_pixels) if particle_pixels.size > 0 else 0
        avg_intensity = np.mean(particle_pixels) if particle_pixels.size > 0 else 0

        # Step 8: Add data to dictionary
        particle_stats[i] = {
            "centroid": centroid,
            "centroidnm": (centroid[0]* pixelWidthnm,centroid[1]*pixelHeightnm),
            "radius": radius*pixelWidthnm,
            "area": area*pixelArea,
            "difference_from_circular": difference_from_circular,
            "max_intensity": max_intensity,
            "average_intensity": avg_intensity,
            "perimeter": perimeter*pixelWidthnm,
            "contours": contours[0]  # Store the main contour for reference
        }

    return particle_stats


def calculate_density_graph(markers, pixelWidthnm, stats, blur_factor=0.1):
    """
    Calculate the local density of particles using Gaussian blur, ensure at least 
    one pixel per particle in marker array, and return a density graph.

    Args:
        marker_array: 2D array where particle markers are placed.
        pixelWidthnm: The width of a pixel in nanometers.
        stats: Particle stats containing area and size information.
        blur_factor: Fraction of the image size to determine the blur size (default 10%).

    Returns:
        None (displays density graph)
    """
    marker_array = np.zeros_like(markers)
    # Ensure at least one pixel per particle
    for i in range(1, len(stats)):
        centroid = stats[i]["centroid"]
        marker_array[int(centroid[1]), int(centroid[0])] = 1

    # Apply Gaussian blur
    blur_size = int(min(marker_array.shape) * blur_factor)
    if blur_size % 2 == 0:
        blur_size += 1
    blurred_density = cv2.GaussianBlur(marker_array.astype(np.float32), (blur_size, blur_size), 0)
    density_map = blurred_density / ((pixelWidthnm/1000)**2)
    #density_histogram, bin_edges = np.histogram(density_map, bins=50, density=True)
    return density_map


def find_nearest_neighbors_angular_optimized(particle_stats, max_distance_nm, angular_step=10):
    """
    Optimized procedure to find the nearest neighbors for each particle in angular steps. 
    First, filters particles by a maximum distance before performing angle calculations.

    Args:
        particle_stats: Dictionary containing particle information with their centroids in nanometers.
        max_distance_nm: The maximum distance (in nm) within which to search for neighbors.
        angular_step: The size of each angular step in degrees (default is 10 degrees).

    Returns:
        nearest_neighbors: A dictionary where each key is the particle index, and the value is a list of 
                           the nearest neighbors in each angular window.
    """
    centroids = np.array([stats["centroidnm"] for stats in particle_stats.values()])
    nearest_neighbors = {}

    # Step 2: Loop through each particle to find its nearest neighbors
    for i, centroid_i in enumerate(centroids):
        # Initialize an empty list to store nearest neighbors in each angular window
        neighbors_in_angles = []

        # Step 3: Calculate the Euclidean distance from the current particle to all other particles
        distances = np.linalg.norm(centroids - centroid_i, axis=1)

        # Step 4: Filter particles that are within the maximum distance
        close_particles = np.where((distances < max_distance_nm) & (distances > 0))[0]
        particle_angles = np.arctan2(centroids[close_particles, 1] - centroid_i[1],
                                     centroids[close_particles, 0] - centroid_i[0])
        #normalize to 0-2pi
        particle_angles[particle_angles < 0] += 2 * np.pi
        
        # Step 5: Loop through angular steps from 0 to 360 degrees
        for angle_deg in range(0, 360, angular_step):
            angle_rad_start = np.radians(angle_deg)
            angle_rad_end = np.radians(angle_deg + angular_step)

            closest_particle = None
            min_distance = np.inf
            
            #get particles in the angular window
            angle_mask = (particle_angles >= angle_rad_start) & (particle_angles < angle_rad_end)
            close_particles_in_angle = close_particles[angle_mask]
            
            for j in close_particles_in_angle:
                distance = distances[j]
                if distance < min_distance:
                    min_distance = distance
                    closest_particle = j

            # Append the closest particle found in this angular window
            if closest_particle is not None:
                neighbors_in_angles.append({
                    "neighbor_index": closest_particle,
                    "distance_nm": min_distance,
                    "angle_deg": angle_deg
                })

        # Store the nearest neighbors for particle i
        nearest_neighbors[i] = neighbors_in_angles

    return nearest_neighbors

def SummaryStats(density_map, particle_stats, neighbors, np_mask, pixelWidthnm, pixelHeightnm):
    flatmap = density_map.flatten()
    _,ax = plt.subplots(1,3,figsize=(3*3,3))
    ax[0].hist(flatmap, bins='auto', density=True)
    ax[0].set_xlabel('Density (um$^{-2}$)')


    imageStats = { 'density_um2': np.mean(flatmap), 'std': np.std(flatmap) }
    imageStats['SimpleDensity_um2'] = len(particle_stats)/(np.prod(np_mask.shape)*pixelWidthnm*pixelHeightnm/1e6)

    del flatmap

    radii = [particle_stats[x]['radius'] for x in particle_stats]
    imageStats['mean_radius_nm'] = np.mean(radii)
    imageStats['radii_map'] = radii
    ax[1].hist(radii,bins='auto',density=True)
    ax[1].set_xlabel('Radius (nm)')
    ax[1].set_ylabel('Count')
    

    closestDistances = []
    for part in neighbors:
        partStats=neighbors[part]
        distances = [ x['distance_nm'] for x in partStats]
        closestDistances.extend(distances)
        
    imageStats['mean_closest_neighbor_distance_nm'] = np.mean(closestDistances)
    imageStats['std_closest_neighbor_distance_nm'] = np.std(closestDistances)

    imageStats['partPerUm2'] = 1/ imageStats['density_um2']
    imageStats['numberPerJunction']= 4*5e-3 * imageStats['partPerUm2'] 
        
    ax[2].hist(closestDistances, bins='auto', density=True)
    ax[2].set_xlabel('Clostest Neighbors Distance (nm)')
    ax[2].set_ylabel('Density')
    plt.show()    
    return imageStats    
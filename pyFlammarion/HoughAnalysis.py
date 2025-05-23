import numpy as np
import matplotlib.pyplot as plt 
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import filters
from skimage.transform import probabilistic_hough_line
from skimage.filters import threshold_multiotsu
import os
#load png image for use with skimage
from skimage import io
from skimage import img_as_ubyte
from skimage import color

import matplotlib.pyplot as plt
from skimage import io, color, exposure, feature, transform


# Preprocessing: Adjust the image contrast
image_eq = exposure.equalize_adapthist(image, clip_limit=0.03)

# Edge Detection: Use the Canny edge detector
edges = feature.canny(image_eq, sigma=.5)

# Hough Transform to detect lines
hough_lines = transform.probabilistic_hough_line(edges, threshold=30, line_length=50, line_gap=3)


from skimage.filters import threshold_mean
#edges = canny(data, 2, 1, 25)
edges = filters.sobel(dataNorm)
thresh = threshold_mean(edges)
image = edges > thresh
plt.imshow(image)

# Classic straight-line Hough transform
# Set a precision of 0.5 degree.
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
h, theta, d = hough_line(image, theta=tested_angles)

dd=hough_line_peaks(h, theta, d)
diffHist={}
for i in range(len(dd[0])):
    diffHist[i]=[]
i=-1    
for _, angle, dist in zip(*dd):
    i+=1
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    slope=np.tan(angle + np.pi/2)
    slope2=np.array([np.cos(angle), np.sin(angle)])
    for x in range(data.shape[0]):
        y=y0+slope*(x-x0)
        if y>0 and y<data.shape[1]:
            x1=int(x+slope2[0]*10)
            y1=int(y+slope2[1]*10)
            
            x2=int(x-slope2[0]*10)
            y2=int(y-slope2[1]*10)
            if x1>0 and x1<data.shape[0]:
                if y1>0 and y1<data.shape[1]:
                    if x2>0 and x2<data.shape[0]:
                        if y2>0 and y2<data.shape[1]:
                            diffHist[i].append((data[x1,y1]-data[x2,y2]))
i=-1                            
for _, angle, dist in zip(*dd):
    i+=1
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    slope=np.tan(angle + np.pi/2)
    slope2=np.array([np.cos(angle), np.sin(angle)])
    for y in range(data.shape[1]):
        x=x0+slope*(y-y0)
        if x>0 and x<data.shape[0]:
            x1=int(x+slope2[0]*10)
            y1=int(y+slope2[1]*10)
            
            x2=int(x-slope2[0]*10)
            y2=int(y-slope2[1]*10)
            if x1>0 and x1<data.shape[0]:
                if y1>0 and y1<data.shape[1]:
                    if x2>0 and x2<data.shape[0]:
                        if y2>0 and y2<data.shape[1]:
                            diffHist[i].append((data[x1,y1]-data[x2,y2]))                            
            

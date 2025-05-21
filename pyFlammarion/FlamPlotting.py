
import numpy as np
import matplotlib.pyplot as plt
from skimage import  segmentation,  measure
from matplotlib.colors import Normalize 
from MathTools import SetPrefix
from FileLoaders.FlammarionFile import FlammarionFile, FlammarionImageData

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

def _plotSegmentation(ax, iWidth,  iHeight , segments, **kwargs):
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
        y_scale = iHeight   / rows
        x_scale = iWidth  / cols
        
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
        y_scale = iHeight  / rows
        x_scale = iWidth  / cols
        
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
    
    
def _SinglePlot(ax:any, imagePack:FlammarionImageData, title:str, mask:np.array, segments:any,show_region_labels:bool,colormap:str, **kwargs):
    
    # Define width and height in micrometers
    iWidth,iHeight = imagePack.physicalSize
    scaleXY, prefixXY = SetPrefix(imagePack.physicalSize )
    
    dataUnit = imagePack.zunit
    image = imagePack.data
    
    scaleZ, prefixZ = SetPrefix(image.ravel())
    
    if title is not None:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'{imagePack.label} - {imagePack.direction}', fontsize=12, fontweight='bold')
        
    # Determine normalization range, excluding outliers
    if 'vrange' in kwargs:
        vmin, vmax = np.percentile(image, kwargs['vrange'])
    else:
        vmin, vmax = np.percentile(image, [1, 99])  # 1st and 99th percentiles
        
    # Display the AFM data
    im = ax.imshow(
        image * scaleZ,
        cmap=colormap,
        extent=[0, iWidth * scaleXY, iHeight * scaleXY, 0],  # Flipped y-axis coordinates
        interpolation='nearest',
        norm=Normalize(vmin=vmin * scaleZ, vmax=vmax * scaleZ),  # Normalize with determined range
        origin='upper'  # Set origin to upper-left corner for top-to-bottom display
    )
    
    # Add a transparent red overlay for masked regions if mask is in kwargs
    if mask is not None:
        # Create a red overlay for masked areas
        mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
        mask_overlay[mask == 0, 0] = 1  # Red channel
        mask_overlay[mask == 0, 3] = 0.5  # Alpha channel (transparency)
        
        # Add the overlay to the image
        ax.imshow(mask_overlay, extent=[0, iWidth * scaleXY, 0, iHeight * scaleXY], 
                    interpolation='nearest', origin='lower')
    
    # Add colored segment outlines if 'segments' is provided
    if segments is not None:
        _plotSegmentation(ax, iWidth*scaleXY, iHeight*scaleXY, segments, show_region_labels=show_region_labels, **kwargs)
        
    
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

def _plotHistogram(ax, imageIndex:int, image: FlammarionImageData,colormap:str, **kwargs):
     
    img_data =image.data
    unit = image.zunit
    
    scale, prefix = SetPrefix(img_data)
    img_data = img_data * scale
    
    # Determine normalization range for consistent coloring with image
    if 'vrange' in kwargs:
        vmin, vmax = np.percentile(img_data, kwargs['vrange'])
    else:
        vmin, vmax = np.percentile(img_data, [1, 99])
    
    # Get the same colormap as used in _SinglePlot
    cmap = plt.cm.get_cmap(colormap)
    
    # Create histogram on the second subplot
    ax[imageIndex].set_title('Height Distribution', fontsize=12, fontweight='bold')

    # Calculate histogram
    hist, bin_edges = np.histogram(img_data, bins=100, range=(vmin, vmax))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize bin heights for visualization
    hist_norm = hist / np.max(hist)

    # Set figure size to match or be slightly smaller than the image subplot
    bbox = ax[0].get_position()
    ax[imageIndex].set_position([bbox.x0 + bbox.width * 1.05, bbox.y0, bbox.width * 0.9, bbox.height])

    # Draw bars with colors from the colormap
    for i, (x, h) in enumerate(zip(bin_centers, hist)):
        # Normalize x to [0, 1] range for colormap
        color_val = (x - vmin) / (vmax - vmin) if vmax > vmin else 0
        color_val = max(0, min(1, color_val))  # Clamp to [0, 1]
        ax[imageIndex].bar(x, hist_norm[i], width=(bin_edges[i+1] - bin_edges[i]), 
                color=cmap(color_val), edgecolor=cmap(color_val), alpha=0.8)

    ax[imageIndex].set_xlabel(f'Height ({prefix}{unit})', fontsize=12, fontweight='bold')
    ax[imageIndex].set_ylabel('Normalized Frequency', fontsize=12, fontweight='bold')
    ax[imageIndex].grid(True, alpha=0.3)
    

def AFMPlot(images:FlammarionFile|FlammarionImageData, show_histogram=False, colormap='afmhot', **kwargs):
    """
    Plot a Flammarion image or a dictionary of images.
    Parameters
    ----------
    image : FlammarionFile or FlammarionImageData
        The image or dictionary of images to plot.
    show_histogram : bool, optional
        Whether to show the histogram of the image data. The default is False.
    **kwargs : keyword arguments
        Additional arguments to pass to the plotting function.
    Returns
    -------
    None
    """
    title = kwargs.pop('title', None)
    mask = kwargs.pop('mask', None)
    segments = kwargs.pop('segments', None)
    show_region_labels = kwargs.pop('show_region_labels', False)
     
    
    showPlot=True
    #if images is a dictionary, extract the image data
    if  isinstance(images, FlammarionFile):
        isMultipleChannels = True
        keys =  images.imageKeys()
        cols = 2
        rows = int(np.ceil(len(keys)/cols))
        fig, ax = plt.subplots(rows, cols, **kwargs)
    elif isinstance(images, FlammarionImageData):
        isMultipleChannels = False
        #check if kwargs already has a figure and axis
        if 'fig' in kwargs.keys() and 'ax' in kwargs.keys():
            fig = kwargs['fig']
            ax = kwargs['ax']
            showPlot=False
        else:
            # Create a new figure and axes
            # If show_histogram is True and this is a single channel image without assigned fig/ax,
            # create a figure with two subplots side by side
            if show_histogram and not isMultipleChannels:
                # Adjust figure size if provided
                if 'figsize' in  kwargs:
                    orig_width, height = kwargs['figsize']
                    kwargs['figsize'] = (orig_width * 2, height)  # Double the width
                
                fig, ax = plt.subplots(1, 2, **kwargs)
            else:
                fig, ax = plt.subplots(1, 1, **kwargs)
    else:
        raise ValueError("images should be a FlammarionFile or FlammarionImageData object")

    #remove ax from kwargs if it exists
    if 'ax' in kwargs.keys():
        del kwargs['ax']
            
    if isMultipleChannels:
        for i, key in enumerate(keys):
            image = images.getImage(key)
            _SinglePlot(ax[i//2, i%2], image, title,mask,segments,show_region_labels, colormap, **kwargs)
    else:
        if show_histogram and not isinstance(ax, plt.Axes):
            # Plot the image on the first subplot
            _SinglePlot(ax[0], images, title,mask,segments,show_region_labels,colormap, **kwargs)
            _plotHistogram(ax, 1, images,colormap, **kwargs)
        else:
            # Regular single plot without histogram
            _SinglePlot(ax, images, title,mask,segments,show_region_labels,colormap,**kwargs)
        
    if showPlot:
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.show()
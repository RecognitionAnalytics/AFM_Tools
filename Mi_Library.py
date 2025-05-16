
    



def smallSlope_planeFlattening(topography, edge_size_percent=0.05, percentile_cutoff=75):
    """
    Perform linear plane flattening of the terraced AFM image with slope filtering.
    
    Parameters:
    -----------
    topography : 2D numpy array
        Height data from AFM measurement
    edge_size_percent : float
        Edge size as a percentage of the whole image for local slope calculation
    percentile_cutoff : float
        Percentile threshold for filtering out high slopes (0-100)
    
    Returns:
    --------
    flattened : 2D numpy array
        Flattened topography data
    """
    edge_size = int(edge_size_percent * min(topography.shape))
    slopes = []
    slope_magnitudes = []
    
    # Calculate local slopes
    for i in range(0, topography.shape[0] - edge_size, edge_size):
        for j in range(0, topography.shape[1] - edge_size, edge_size):
            local_patch = topography[i:i + edge_size, j:j + edge_size]
            x = np.arange(local_patch.shape[1])
            y = np.arange(local_patch.shape[0])
            X, Y = np.meshgrid(x, y)
            X = X.ravel()
            Y = Y.ravel()
            Z = local_patch.ravel()
            
            A = np.c_[X, Y, np.ones(X.shape)]
            C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
            slopes.append(C[:2])
            # Calculate magnitude of slope vector
            slope_magnitudes.append(np.sqrt(C[0]**2 + C[1]**2))
    
    slopes = np.array(slopes)
    slope_magnitudes = np.array(slope_magnitudes)
    
    # Filter out high slopes
    magnitude_threshold = np.percentile(slope_magnitudes, percentile_cutoff)
    filtered_slopes = slopes[slope_magnitudes <= magnitude_threshold]
    
    # If no slopes pass the filter, gradually increase threshold
    while len(filtered_slopes) == 0 and percentile_cutoff < 100:
        percentile_cutoff += 5
        magnitude_threshold = np.percentile(slope_magnitudes, percentile_cutoff)
        filtered_slopes = slopes[slope_magnitudes <= magnitude_threshold]
    
    # Use median of filtered slopes
    most_likely_slope = np.median(filtered_slopes, axis=0)
    
    # Create and subtract plane
    X, Y = np.meshgrid(np.arange(topography.shape[1]), np.arange(topography.shape[0]))
    plane = most_likely_slope[0] * X + most_likely_slope[1] * Y
    
    flattened = topography - plane
    return flattened
 
def localSlope_planeFlattening(topography, edge_size_percent=0.05):
    """
    Perform linear plane flattening of the terraced AFM image.
    
    Parameters:
    -----------
    topography : 2D numpy array
        Height data from AFM measurement
    edge_size_percent : float
        Edge size as a percentage of the whole image for local slope calculation
    
    Returns:
    --------
    flattened : 2D numpy array
        Flattened topography data
    """
    edge_size = int(edge_size_percent * min(topography.shape))
    slopes = []
    
    for i in range(0, topography.shape[0] - edge_size, edge_size):
        for j in range(0, topography.shape[1] - edge_size, edge_size):
            local_patch = topography[i:i + edge_size, j:j + edge_size]
            x = np.arange(local_patch.shape[1])
            y = np.arange(local_patch.shape[0])
            X, Y = np.meshgrid(x, y)
            X = X.ravel()
            Y = Y.ravel()
            Z = local_patch.ravel()
            
            A = np.c_[X, Y, np.ones(X.shape)]
            C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
            slopes.append(C[:2])
    
    slopes = np.array(slopes)
    most_likely_slope = np.median(slopes, axis=0)
    
    X, Y = np.meshgrid(np.arange(topography.shape[1]), np.arange(topography.shape[0]))
    plane = most_likely_slope[0] * X + most_likely_slope[1] * Y
    
    flattened = topography - plane
    return flattened


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

def ClusterSurface(flat, maxTerrace_n=5):
    # Prepare the data for clustering
    flat_1d = flat.ravel().reshape(-1, 1)  # Flatten the AFM data

    # Determine the number of terraces using Gaussian Mixture Model (GMM)
    bic_scores = []
    n_components_range = range(1, maxTerrace_n)  # Test 1 to 10 components
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(flat_1d)
        bic_scores.append(gmm.bic(flat_1d))

    # Optimal number of clusters (terraces) is the one with the lowest BIC
    optimal_n_components = n_components_range[np.argmin(bic_scores)]
    gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
    gmm.fit(flat_1d)
    labels = gmm.predict(flat_1d)

    # Get cluster statistics
    cluster_stats = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_heights = flat_1d[cluster_mask]
        avg_height = np.mean(cluster_heights)
        std_dev = np.std(cluster_heights)
        rms_roughness = np.sqrt(np.mean(cluster_heights**2))
        area_fraction = np.sum(cluster_mask) / flat_1d.size * 100  # Percentage of total area
        #(label, avg_height, std_dev, rms_roughness, area_fraction)
        cluster_stats.append({ 'label_mask':cluster_mask .reshape(flat.shape), 'avg_height':avg_height, 'std_dev':std_dev, 'rms_roughness':rms_roughness, 'area_fraction':area_fraction})    
        
    return labels.reshape(flat.shape), cluster_stats

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


   
    

def FindPlateaus(data, num_plateaus=3, plot=True):
    """
    Find plateaus in the data using multi-Otsu thresholding.

    Parameters:
    - data: 2D numpy array of image data.
    - num_plateaus: Number of plateaus to find.
    - plot: Boolean indicating whether to plot the results.

    Returns:
    - thresholds: List of threshold values.
    """
        # Using multi-Otsu to find the thresholds.
    thresholds = threshold_multiotsu(data,num_plateaus)

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(data, bins=thresholds)
    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

        # Plotting the original image.
        ax[0].imshow(data, cmap='gray')
        ax[0].set_title('Original')
        ax[0].axis('off')

        # Plotting the histogram and the two thresholds obtained from
        # multi-Otsu.
        ax[1].hist(data.ravel(), bins=255)
        ax[1].set_title('Histogram')
        for thresh in thresholds:
            ax[1].axvline(thresh, color='r')

        # Plotting the Multi Otsu result.
        ax[2].imshow(regions, cmap='jet')
        ax[2].set_title('Multi-Otsu result')
        ax[2].axis('off')
        plt.subplots_adjust()
        plt.show()
    
    return regions, thresholds    

def StepFlatten(data,verbose=False):
    #determine coordinates
    X, Y = np.meshgrid(np.linspace(0, 1, data.shape[0]),  np.linspace(0, 1, data.shape[1]), copy=False)
    X = X.flatten()
    Y = Y.flatten()

    #iteration variable
    ravelData=np.ravel(data)
    flattenedData =np.zeros_like(ravelData)

    #calculate all the terms for third order plane fit
    terms2 = [X*0+1, X, Y, X**2, Y**2, X**3, Y**3,X*Y,X**2*Y,Y**2*X,X**2*Y**2,X**3*Y, Y**3*X ,X**3*Y**2, Y**3*X**2, Y**3*X**3]
    A = np.array(terms2).T
    for repeat in range(10):

        #the histogram allows us to segment the image into planes
        m=np.min(ravelData)
        M=np.max(ravelData)
        r=M-m
        m=m-r*.25
        M=M+r*.25

        bins=np.arange(m,M,(M-m)/100)

        #meanshift is a 2D routine, but our data is 1D in height
        ff=ravelData.reshape(-1, 1)
        bandwidth = estimate_bandwidth(ff, quantile=0.2,  n_samples=1000)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(ff)

        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        for k in labels_unique:
            my_members = labels == k
            mean = np.mean(ravelData[my_members])
            flattenedData[my_members]=ravelData[my_members]-mean
        if verbose:
            plt.imshow(np.reshape( flattenedData, newshape=[data.shape[0],data.shape[1]]), cmap='gray')
            plt.show()

        coeff, r, rank, s = np.linalg.lstsq(A,flattenedData,rcond=None)
        p=coeff[0]*terms2[0]
        for i in range(1,len(terms2)):
            p+=coeff[i]*terms2[i]

        ravelData=ravelData-p
        if verbose:
            plt.imshow(np.reshape( ravelData, newshape=[data.shape[0],data.shape[1]]), cmap='gray')
            plt.show()    
    
    minData=np.min([ np.mean( ravelData[labels==i]  ) for i in range(np.max(labels))])
    ravelData=ravelData-minData
    return np.reshape( ravelData, newshape=[data.shape[0],data.shape[1]]),np.reshape(labels, newshape=[data.shape[0],data.shape[1]]), np.histogram(ravelData,bins=50)
    
def extractBackground(my_members, selectedData, terms2,verbose):
    lTerms =[]
    for x in terms2:
        lTerms.append(x[my_members])

    A = np.array(lTerms).T
    coeff, r, rank, s = np.linalg.lstsq(A,selectedData,rcond=None)

    if verbose:
        testImage =np.zeros([256,256],dtype=np.float32)
        for i in range(len(lTerms[0])):
            testImage[int(255*lTerms[2][i]), int(lTerms[1][i]*255) ]=selectedData[i]
        plt.imshow(testImage,cmap="gray")
        plt.show()
    return coeff

def IterFlatten(data,backgroundSelect=0,verbose=False):

    #determine coordinates
    X, Y = np.meshgrid(np.linspace(0, 1, data.shape[0]),  np.linspace(0, 1, data.shape[1]), copy=False)
    X = X.flatten()
    Y = Y.flatten()

    #iteration variable
    flattenedData=np.ravel(data)

    #calculate all the terms for third order plane fit
    terms2 = [X*0+1, X, Y, X**2, Y**2, X**3, Y**3,X*Y,X**2*Y,Y**2*X,X**2*Y**2,X**3*Y, Y**3*X ,X**3*Y**2, Y**3*X**2, Y**3*X**3]


    for repeat in range(10):
        
        #the histogram allows us to segment the image into planes
        m=np.min(flattenedData)
        M=np.max(flattenedData)
        r=M-m
        m=m-r*.25
        M=M+r*.25

        bins=np.arange(m,M,(M-m)/100)

        #meanshift is a 2D routine, but our data is 1D in height
        ff=flattenedData.reshape(-1, 1)
        bandwidth = estimate_bandwidth(ff, quantile=0.2,  n_samples=1000)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(ff)

        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        
        clusterValue = {}
        clusterSize ={}
        for k in labels_unique:
            my_members = labels == k
            selectedData = flattenedData[my_members]
            if (len(selectedData)>800):
                clusterValue[k] = np.mean(selectedData)
                clusterSize[k] = len(selectedData)
        sorted_byValue = sorted(clusterValue.items(), key=operator.itemgetter(1))
        sorted_bySize = sorted(clusterSize.items(), key=operator.itemgetter(1))
        
        if backgroundSelect==0:
            for k in sorted_byValue:
                my_members = (labels == k[0])
                selectedData = flattenedData[my_members]
                if len(selectedData)>1000:
                    coeff=extractBackground(my_members, selectedData, terms2,verbose)
                    break
        elif backgroundSelect==1:
            k=sorted_byValue[-1][0]
            my_members = (labels == k)
            selectedData = flattenedData[my_members]
            coeff=extractBackground(my_members, selectedData, terms2,verbose)
        elif backgroundSelect==2:
            k=sorted_bySize[-1][0]
            my_members = (labels == k)
            selectedData = flattenedData[my_members]
            coeff=extractBackground(my_members, selectedData, terms2,verbose)
        else:
            coeff=[]
            cc=0
            for k in range(0,n_clusters_):
                my_members = labels == k
                selectedData = flattenedData[my_members]
                x=terms2[1][my_members]
                y=terms2[2][my_members]
                f=len(selectedData)*np.std(x)*np.std(y)
                if len(selectedData)>1000:
                    cs=extractBackground(my_members, selectedData, terms2,verbose)
                    if (len(coeff)==0):
                        coeff=cs*f
                    else:
                        coeff+=cs*f
                    cc+=f
            coeff=np.array(coeff)/cc

        
            
                

        #calculate the background plane
        p=coeff[0]*terms2[0]
        for i in range(1,len(terms2)):
            p+=coeff[i]*terms2[i]
            
        flattenedData -= p
        
        if verbose:
            plt.imshow(np.reshape( flattenedData, newshape=[data.shape[0],data.shape[0]]), cmap='gray')
            plt.show()

            plt.hist(flattenedData, bins)
            plt.show()
            
    ff=flattenedData.reshape(-1, 1)
    bandwidth = estimate_bandwidth(ff, quantile=0.2,  n_samples=1000)            
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(ff)

    labels = ms.labels_

    return np.reshape(flattenedData, newshape=[data.shape[0],data.shape[0]]), np.reshape(labels, newshape=[data.shape[0],data.shape[0]]), np.histogram(flattenedData,bins)

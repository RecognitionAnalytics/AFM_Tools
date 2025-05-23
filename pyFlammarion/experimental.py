
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

def CreateTexture(data,minD,maxD,file, width,height):
    dataNorm=((data-minD)/(maxD-minD))*255
    dataNorm[dataNorm<50]=50
    dataNorm[0,0]=0
    dataNorm[dataNorm>255]=255

    fig = plt.figure()
    plt.imshow(dataNorm, cmap=plt.get_cmap("gist_heat"))
    plt.axis('off')
    

    textureFile =folderName + "\\processed\\" + file + "_texture.jpg"
    plt.savefig( textureFile, bbox_inches='tight', pad_inches = 0)
    plt.show()
    
    fig = plt.figure()
    if width>1000 or height>1000:
        unit="µm"
        width=width/1000
        height=height/1000
    else:
        unit ="nm"
        
        
    dataNorm=data+.0001
    dataNorm[dataNorm<minD]=minD
    dataNorm[dataNorm>maxD]=maxD
    
    shw=plt.imshow(dataNorm, cmap=plt.get_cmap("gist_heat"),extent=(0, width,0, height))
    bar = plt.colorbar(shw)
    # show plot with labels
    plt.xlabel(unit)
    plt.ylabel(unit)
    bar.set_label('Height (nm)')
    
    textureFile =folderName + "\\processed\\" + file + "_drawn.jpg"
    plt.savefig( textureFile, bbox_inches='tight', pad_inches = 0)
    plt.show()
    
    
    
def CreateHistogram(data,minD,maxD,file, imageArea):
    bins=50#np.arange(minD,maxD,(maxD-minD)/100)
    hFlat = np.histogram(np.ravel(data),bins=bins)
    bins=hFlat[1][:-1]
    vals=hFlat[0]/np.sum(hFlat[0])*imageArea/1000/1000
    b=plt.bar( bins,(vals),width=1.0)
    plt.xlabel("Height (nm)")
    plt.ylabel("Area (um^2)")
    histFile = folderName + "\\processed\\" + file + "_hist.svg"
    plt.savefig(histFile , bbox_inches='tight', pad_inches = 0)
    plt.show()

def CreateProfileLines(data,minD,maxD,Xrange,file)   :    
    bestHLine=0
    bestHStd=0
    smoothHLine=0
    smoothHStd=1e10

    bestVLine=0
    bestVStd=0
    smoothVLine=0
    smoothVStd=1e10

    maxH=0
    maxHV=0
    for i in range(data.shape[0]):
        lineS = np.std( data[i,:])
        
        if lineS>maxHV:
            maxHV=lineS
            maxH=i
            
        
    maxV=np.argmax(data[maxH,:])

    for i in range(4,data.shape[0]-4):
        if np.abs(i-maxH)>10:
            lineS = np.std( data[i,:])

            if lineS>bestHStd:
                bestHStd=lineS
                bestHLine=i
            if lineS<smoothHStd:
                smoothHStd=lineS
                smoothHLine=i

    for i in range(4,data.shape[0]-4):
        if np.abs(i-maxV)>10:
            lineS = np.std( data[:,i])
            if lineS>bestVStd:
                bestVStd=lineS
                bestVLine=i
            if lineS<smoothVStd:
                smoothVStd=lineS
                smoothVLine=i
    
    profileLines={ "horizontal":[data[bestHLine,:].tolist(),    data[smoothHLine,:].tolist()], "vertical":[data[:,bestVLine].tolist(),    data[:,smoothVLine].tolist()]}
    
    dataNorm=((data-minD)/(maxD-minD))*255
    dataNorm[dataNorm<0]=0
    dataNorm[0,0]=0
    dataNorm[dataNorm>255]=255

    fig, ax1 = plt.subplots()
    unit='nm'
    maxDistance=Xrange[0] #in nm
    step=Xrange[0]/len(profileLines["horizontal"][0])
    if maxDistance>1000:
        maxDistance/=1000
        step/=1000
        unit='uM'
           
    ax1.plot(np.arange(0,maxDistance,step) , profileLines["horizontal"][0],color='blue')
    ax1.plot(np.arange(0,maxDistance,step),profileLines["horizontal"][1],color='black')
    ax1.set_xlabel('Distance (' + unit + ')')
    ax1.set_ylabel('Height (nm)')
    ax2 = plt.axes([0,0,1,1])
    ip = InsetPosition(ax1, [0.01,.6,0.3,0.3])
    ax2.set_axes_locator(ip)
    ax2.imshow(dataNorm, cmap=plt.get_cmap("gist_heat"))
    ax2.axis('off')
    ax2.axhline(y=bestHLine-1,linewidth=3,color='cornflowerblue')
    ax2.axhline(y=smoothHLine-1,linewidth=3,color='gray')
    
    ax2.axhline(y=bestHLine,color='blue')
    ax2.axhline(y=smoothHLine,color='black')
    file = os.path.basename(filename).replace(".npy","")
    textureFile =folderName + "\\processed\\" + file + "_hprofile.svg"
    plt.savefig( textureFile, bbox_inches='tight', pad_inches = 0)
    plt.show()        
    
    fig, ax1 = plt.subplots()
    unit='nm'
    maxDistance=Xrange[1] #in nm
    step=Xrange[1]/len(profileLines["horizontal"][0])
    if maxDistance>1000:
        maxDistance/=1000
        step/=1000
        unit='uM'
    ax1.plot(np.arange(0,maxDistance,step) ,profileLines["vertical"][0],color='blue')
    ax1.plot(np.arange(0,maxDistance,step) ,profileLines["vertical"][1],color='black')
    ax1.set_xlabel('Distance (' + unit + ')')
    ax1.set_ylabel('Height (nm)')
    ax2 = plt.axes([0,0,1,1])
    ip = InsetPosition(ax1, [0.01,.6,0.3,0.3])
    ax2.set_axes_locator(ip)
    ax2.imshow(dataNorm, cmap=plt.get_cmap("gist_heat"))
    ax2.axis('off')
    ax2.axvline(x=bestVLine-1,linewidth=3,color='cornflowerblue')
    ax2.axvline(x=smoothVLine-1,linewidth=3,color='gray')
    ax2.axvline(x=bestVLine,color='blue')
    ax2.axvline(x=smoothVLine,color='black')
    file = os.path.basename(filename).replace(".npy","")
    textureFile =folderName + "\\processed\\" + file + "_vprofile.svg"
    plt.savefig( textureFile, bbox_inches='tight', pad_inches = 0)
    plt.show()    
    
    return profileLines

def CreateStats(data,labels,pixelArea):
    stats=[]

    rData=np.ravel(data)    
    lData =np.ravel(labels)
    minD=np.min(rData)
    maxD=np.max(rData)
    meanD=np.mean(rData)
    stdD=np.std(rData)
    roughD =np.sum( np.abs(rData-meanD))/len(rData)
    
    unit="pm"
    rFactor=1000
    if (stdD>1):
        unit="nm"
        rFactor=1
        
        
    areaUnit ="nm^2"    
    area =pixelArea*data.shape[0]*data.shape[1]
    if area>1:
        area=area/1000/1000
        areaUnit = "um^2"
    print(peaks)
    stats.append({
            "Name":"Whole Image",
            "Average value (nm)":meanD,
            "RMS roughness (" + unit +")":stdD*rFactor,
            "Mean roughness (" + unit +")":roughD*rFactor,
            "Skew":scipy.stats.skew(rData),
            "Maximum (nm)":maxD,
            "Minimum (nm)":minD,
            "Area (" + areaUnit + ")": area
        })
    
    
    labels_unique = np.unique(labels)
        
    for k in labels_unique:
        my_members = lData == k
        selectedData = rData[my_members]

        minD=np.min(selectedData)
        maxD=np.max(selectedData)
        meanD=np.mean(selectedData)
        stdD=np.std(selectedData)
        roughD =np.sum( np.abs(selectedData-meanD))/len(rData)
        unit="pm"
        rFactor=1000
        if (stdD>1):
            unit="nm"
            rFactor=1
        areaUnit ="nm^2"    
        area =len(selectedData)*pixelArea
        if area>1000:
            area=area/1000/1000
            areaUnit = "um^2"
        
        stats.append({
                "Name":"layer " + str(k),
                "Average value (nm)":meanD,
                "RMS roughness (" + unit +")":stdD*rFactor,
                "Mean roughness (" + unit +")":roughD*rFactor,
                "Skew":scipy.stats.skew(selectedData),
                "Maximum (nm)":maxD,
                "Minimum (nm)":minD,
                "Area (" + areaUnit + ")": area
            })
        
    return stats

def GetBlobs(data,minD,maxD, dStats, addCol):
    # Read image
    im=((data-minD)/(maxD-minD))*255
    im[im<0]=0
    im[0,0]=0
    im[im>255]=255
    im = im.astype(np.uint8)


    image_gray = im
    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)


    widths=[]
    heights=[]
    for blob in blobs_doh:
        y, x, r = blob
        widths.append(r)
        x2=int(x+2*r)
        y2=int(y+2*r)
        if x2<data.shape[0] and y2<data.shape[1]:
            heights.append(data[int(x),int(y)]-data[x2,y2])
            
    nBlobs =len(widths)
    blobRadius=np.mean(widths)
    blobHeight=np.mean(heights)
    if np.isnan(nBlobs): nBlobs=0
    if np.isnan(blobRadius): blobRadius=0
    if np.isnan(blobHeight): blobHeight=0
        
    if len(widths)>0:
        dStats[addCol]["Number Blobs"]=nBlobs
        dStats[addCol]["Ave Blob Radius (nm)"]=blobRadius
        dStats[addCol]["Ave Blob Height (nm)"]=blobHeight
    return dStats 



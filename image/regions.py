"""Module for image segmentation procedures"""

import numpy
np = numpy
import scipy
import pymorph

import image
import thresholding

# ---
def seeds(img,smooth=3,border=3):
    """
    Returns an array with the seeds identified
    
    Input:
     - img  ndarray : Image array (ndim=2)
     - smooth   int : Size of Gaussian sigma (smoothing)
     - border   int : Size of border; where seeds are regested
    
    Output:
     - seeds : Same 'img' size array with labeled seed points

    ---
    """
    ndi = scipy.ndimage;
    np = numpy;
    
    # Take image maxima
    maxs = image.local_maxima( ndi.gaussian_filter(img,smooth) )

    # Eumarate (label) them
    maxs,nmax = ndi.label(maxs)

    # Remove maxima found near borders
    if border:
        brd = border
        border = np.zeros(img.shape,np.bool)
        border[-brd:,:]=border[:brd,:]=border[:,-brd:]=border[:,:brd] = 1
        for id in np.unique(maxs[border]):
            maxs[maxs==id] = 0

    # Take the seeds (x_o,y_o points)
    seeds = np.zeros(img.shape,np.uint)
    for i,id in enumerate(np.unique(maxs)):
        seeds_tmp = seeds*0
        seeds_tmp[maxs==id] = 1
        ym,xm = ndi.center_of_mass(seeds_tmp)
        seeds[ym,xm] = i
    
    return seeds

# ---
def threshold(img,thresh=0,size=9):
    """Segment using a thresholding algorithm
    
    Input:
     - img  ndarray : Image array (ndim=2)
     - thresh float : Threshold value for pixels selectino
     - size     int : Minimum size a group of pixels must have
    
    Output:
     - regions : Labeled array for each segmented region
    
    ---
    """
    np = numpy;
    ndi = scipy.ndimage;

    if thresh == 0:
        thresh = thresholding.histmax(img)
        thresh += np.std(img)
    
    # Take the binary image version "splitted" on the 'thresh' value
    img_bin = (img > thresh)

    # And use (MO) binary opening (erosion + dilation) for cleaning spurious Trues
    strct = ndi.generate_binary_structure(2,1)
    img_bin = ndi.binary_opening(img_bin,strct)

    # Label each group (Regions==True) of pixels
    regions,nlbl = ndi.label(img_bin)
    for i in xrange(1,nlbl+1):
        inds = np.where(regions==i)
        if inds[0].size < size:
            regions[inds] = 0

    # Reorder regions identities
#    for i,lbl in enumerate(np.unique(regions)):
#        regions[regions==lbl] = i

    return regions.astype(np.bool)

# ---
def region_grow(img,x,y,thresh=0):
    """
    Segment using a Region Growing technique
    
    Input:
     - img  ndarray : Image array (ndim=2)
     - x        int : Seed x position
     - y        int : Seed y position
     - thresh float : Threshold value for limiting the grow
    
    Output:
     - region  ndarray(bool) : Region grown around given 'x,y'
    
    ---
    """
    np = numpy;
    ndi = scipy.ndimage;
    
    flag = 1
    x_o = x
    y_o = y
    
    if thresh == 0:
        thresh = thresholding.histmax(img)
        thresh += np.std(img)

    # Initialize region with the seed point
    region = np.zeros(img.shape,dtype=np.bool)
    reg_old = (region==flag)

    if (img[y_o,x_o] < thresh): return region
    
    region[y_o,x_o] = flag
    reg_cur = (region==flag)

    # For future morphological operations (MO)
    strct_elem = ndi.generate_binary_structure(2,2)

    # While region stills changes (grow), do...
    while not np.all(region == reg_old):
        
        reg_old = region.copy()
        reg_mean = np.mean(img[region==flag])
        #reg_area = np.where(region==flag)[0].size

        # Define pixel neighbours using MO dilation
        tmp = ndi.binary_dilation(region,strct_elem, 1)
        neigbors = tmp - region
        inds = np.where(neigbors)

        # Check for the new neighbors; do they fullfil requirements
        # to become members of the growing region?
        for y_i,x_i in zip(*inds):
            if (img[y_i,x_i] >= thresh):# and (img[y_i,x_i] <= reg_mean):
                region[y_i,x_i] = flag


    return region

# ---
def centroid2id(segimg, centroid):
    """Determine ID of the objects which contains the given pixel.

    Input:
     - segimg : ndarray(ndim=2,dtype=int)
       ndarray with a segmented image (int), e.g, SE's SEGMENTATION 
     - centroid : (int,int)
       List of tuples with position to search for object (x0,y0)

    Output:
     - object ID : int
       List of IDs (int) for each given (x,y) points
       
       
    Example:
    >>> x = 1267
    >>> y = 851
    >>> centroid = zip(x,y)
    >>> objIDs = centroid2ID(SegImg_array,centroid)
    >>> objIDs
    2
    >>> 
    
    """

    # Is there an identified object on given (xo,yo) point? If yes, store its ID..
    #
    xo,yo = centroid;
    xo = int(float(xo));
    yo = int(float(yo));
    objid = segimg[yo,xo];

    return (objid);

# ---
def readout_ids(segimg):
    """ Read segmented image values as object IDs.
    
    Input:
     - segimg : ndarray
        Segmentation image array
    
    Output:
     -> list with object IDs : [int,]
        List with (typically integers) object IDs in 'segimg'
        Value "0" is taken off from output list ID
    
    """

    objIDs = list(set(seg_img.flatten()) - set([0]))

    return objIDs

# ---
def obj_id2index_array(segimg, objID):
    """ 
    Generate mask for each object in given image.
    
    ID in 'objIDs' is used to create a mask for each object (ID) in 'segimg'.
    
    Input:
     - segimg : ndarray(ndim=2,dtype=int)
        Image array with int numbers as object identifiers
     - objIDs : [int,]
        List with IDs for objects inside 'segimg'
        
    Output:
     -> index array (output from numpy.where())
        List of tuples with index arrays in it. The output is a list of "numpy.where()" arrays.
        
    """


    # For each object (id) scan the respective indices for image mask and 'cutout' params
    #
    ide = float(objID);
    mask = np.where(segimg == int(ide));
    
    return mask;

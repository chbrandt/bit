"""Module for image segmentation procedures"""

import numpy
np = numpy
import scipy
from scipy import ndimage as ndi


class Momenta:
    @staticmethod
    def center_of_mass(img):
        return ndi.center_of_mass(img)


class Clean:
    @staticmethod
    def border(img,maxs,size):
        border = np.zeros(img.shape,np.bool)
        border[-size:,:] = 1
        border[:size,:] = 1
        border[:,:size] = 1
        border[:,-size:] = 1
        for id in np.unique(maxs[border]):
            maxs[maxs==id] = 0
        return maxs

    @staticmethod
    def spurious(img_bin):
        # And use (MO) binary opening (erosion + dilation) for cleaning spurious Trues
        strct = ndi.generate_binary_structure(2,1)
        img_bin = ndi.binary_opening(img_bin,strct)
        return img_bin

    @staticmethod
    def small(img_bin,size=9):
        # Label each group (Regions==True) of pixels
        regions,nlbl = ndi.label(img_bin)
        for i in xrange(1,nlbl+1):
            inds = np.where(regions==i)
            if inds[0].size < size:
                regions[inds] = 0
        return regions.astype(np.bool)


# ---
def maxima(img):
    """
    Returns image local maxima
    
    Input:
     - img  ndarray : Image array (ndim=2,dtype=float|int)
    
    Output:
     - maxs : Same 'img' size array with maxima = True (dtype=bool)
    
    ---
    """
    import pymorph
    from image import Transf

    # Image has to be 'int' [0:255] to use in 'pymorph'
    img = Transf.float2uint(img)

    # Search for image maxima
    maxs = pymorph.regmax(img)

    # A closing step is used "clue" near maxima
    elem_strct = ndi.generate_binary_structure(2,2)
    maxs = ndi.binary_closing(maxs,elem_strct)
    return maxs

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
    # Find image maxima
    smoothed_img = ndi.gaussian_filter(img,smooth)
    maxs = maxima(smoothed_img)
    del smoothed_img

    # Label the maxima to properly clean them after
    maxs,nmax = ndi.label(maxs)

    # Remove maxima found near borders
    if border:
        maxs = Clean.borderRegions(img,maxs,border)

    # Take the seeds (x_o,y_o points)
    seeds = np.zeros(img.shape,np.uint)
    for i,id in enumerate(np.unique(maxs)):
        seeds_tmp = seeds*0
        seeds_tmp[maxs==id] = 1
        ym,xm = Momenta.center_of_mass(seeds_tmp)
        seeds[ym,xm] = i
    
    return seeds

# ---
def mask(segimg, objID):
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
    id = float(objID);
    mask = np.where(segimg == int(id));
    return mask;

# ---
def read_labels(segimg):
    """ Read segmented image values as object IDs.
    
    Input:
     - segimg : ndarray
        Segmentation image array
    
    Output:
     -> list with object IDs : [int,]
        List with (typically integers) object IDs in 'segimg'
        Value "0" is taken off from output list ID
    
    """

    objIDs = list(set(segimg.flatten()) - set([0]))
    return objIDs


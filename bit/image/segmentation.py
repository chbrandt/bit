#!/usr/bin/env python

"""Module to deal with objects identification in segmented images"""

##@ segobjs
#
#
# This package contains functions to deal with image objects segmentation.
# Functions are designed primarily to work with Sextractor
#
# Executable package : No

from image import Stats
from regions import Clean
import numpy as np
from scipy import ndimage as ndi

# ---
def threshold(img,thresh=0,size=9):
    """
    Segment image using a thresholding algorithm
    
    Input:
     - img  ndarray : Image array (ndim=2)
     - thresh float : Threshold value for pixels selectino
     - size     int : Minimum size a group of pixels must have
    
    Output:
     - regions : Labeled array for each segmented region
    """

    if thresh == 0:
        thresh = Stats.mode(img)
        thresh += np.std(img)
    
    # Take the binary image version "splitted" on the 'thresh' value
    img_bin = (img > thresh)

    # Clean fluctuations
    img_bin = Clean.spurious(img_bin)

    # Clean small objects
    img_bin = Clean.small(img_bin,size=9)

    return img_bin

# ---
def regionGrow(img,x,y,thresh=0):
    """
    Segment using a Region Growing technique
    
    Input:
     - img  ndarray : Image array (ndim=2)
     - x        int : Seed x position
     - y        int : Seed y position
     - thresh float : Threshold value for limiting the grow
    
    Output:
     - region  ndarray(bool) : Region grown around given 'x,y'
    """

    flag = 1
    x_o = x
    y_o = y
    
    if thresh == 0:
        thresh = Stats.mode(img)
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


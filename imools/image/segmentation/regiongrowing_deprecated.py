import sys

import numpy as np
from scipy import ndimage
import pymorph as pm

def threshold(img,thresh=0,size=9):
    """Segment using a thresholding algorithm
    
    Input:
     - img  ndarray : Image array (ndim=2)
     - thresh float : Threshold value for pixels selectino
     - size     int : Minimum size a group of pixels must have
    
    Output:
     - groupx  ndarray : Group of pixels 
    
    """

    if thresh == 0:
        thresh = np.median(img)/2.
    
    img_thrsh = (img > thresh)

    strct = ndimage.generate_binary_structure(2,1)
    img_thrsh = ndimage.binary_opening(img_thrsh,strct)
    
    img_lbl,nlbl = ndimage.label(img_thrsh)
    for i in range(1,nlbl+1):
        inds = np.where(img_lbl==i)
        if inds[0].size < size:
            img_thrsh[inds] = 0

    return img_thrsh


def region_grow(img,x,y,thresh=0):
    """Segment using a Region Growing technique
    
    Input:
     - img  ndarray : Image array (ndim=2)
     - x        int : Seed x position
     - y        int : Seed y position
     - thresh float : Threshold value for limiting the grow
    
    Output:
     - region  ndarray : Region grown around given 'x,y'
    
    ---
    """
    
    flag = 1
    x_o = x
    y_o = y
    
    if thresh == 0:
        thresh = img[y_o,x_o]/5.

    # Initialize region with the seed point
    #
    region = np.zeros(img.shape,dtype=np.bool)
    reg_old = (region==flag)
    region[y_o,x_o] = flag
    reg_cur = (region==flag)

    # For future morphological operation (MO); use a (3,3)-True element
    #
    strct_elem = ndimage.generate_binary_structure(2,2)

    # While region stills changes (grow), do...
    #
    while not np.all(region == reg_old):
        
        #reg_area = np.where(region==flag)[0].size
        reg_mean = np.mean(img[region==flag])

        # Define pixel neighbours using MO dilation
        #
        tmp = ndimage.binary_dilation(region,strct_elem, 1)
        neigbors = tmp - region
        inds = np.where(neigbors)

        reg_old = region.copy()
        
        # Check for the new neighbors; do they fullfil requirements
        # to become members of the growing region?
        #
        for y_i,x_i in zip(*inds):
            if (img[y_i,x_i] > thresh) and (img[y_i,x_i] <= reg_mean):
                region[y_i,x_i] = flag

    
    return region

# -*- coding: utf-8 -*-

"""
Module to deal with objects identification in segmented images
"""

from image import Stats
import numpy as np
from scipy import ndimage as ndi


def seeds(img,smooth=3,border=3):
    """
    Returns an array with the seeds identified
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


def threshold(img, thresh=None, size=9):
    """
    Segment image using a thresholding algorithm
    """
    if thresh is None:
        thresh = Stats.mode(img)
        thresh += np.std(img)

    # Take the binary image version "splitted" on the 'thresh' value
    img_bin = (img > thresh)
    
    # Clean fluctuations
    img_bin = Clean.spurious(img_bin)
    
    # Clean small objects
    img_bin = Clean.small(img_bin,size=9)
    
    return img_bin


def regionGrow(img, x, y, thresh=None):
    """
    Segment using a Region Growing technique
    """

    flag = 1
    x_o = x
    y_o = y
    
    if thresh is None:
        thresh = Stats.mode(img)
        thresh += np.std(img)

    # Initialize region with the seed point
    region = np.zeros(img.shape,dtype=np.bool)
    reg_old = (region==flag)

    if img[y_o,x_o] < thresh:
        return region
    
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

import numpy as np
from scipy import ndimage as ndi


class Cleaner:
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


class Regions:
    """
    Functions to deal with labeled (or simply binary) regions on images.
    
    Typically, first-stage result of a segmentation process is a binary version
    of the image where 1's flag foreground items and 0's flag the background.
    Pixels flagged with 1's are called regions at this stage.
    """
    def maxima(img):
        """
        Return image local maxima
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
    
    
    def mask(segimg, objID):
        """ 
        Generate mask for each object in given image.
        """
        # For each object (id) scan the respective indices for image mask and 'cutout' params
        id = float(objID);
        mask = np.where(segimg == int(id));
        return mask;
    
    def labels(segimg):
        """
        Read segmented image values as object IDs.
        """
        objIDs = list(set(segimg.flatten()) - set([0]))
        return objIDs
    

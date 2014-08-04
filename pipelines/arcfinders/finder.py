#!/usr/bin/env python
"""Semi-automated segmentation and check procedure, for ArcFinders work"""

# -------------------------------------------------------------------------------------
#
# Copyright (C) 2011 - Carlos Brandt
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
# Carlos Brandt - carloshenriquebrandt@gmail.com
# Nov/2011
#======================================================================================

# Algorithm for semi-automate segmentation
# 1) Smooth image
# 2) Segment image: region-growing (RG) or thresholding (Th)
# - RG: find seeds and segment for each seed above 'thresh_val'
# - Th: Read out regions with values above 'thresh_val' and bigger than 9
# 3) Output centroids
# 4) Match detection if needed/wanted


import logging;
logging.basicConfig(level=logging.INFO,filename='Log_finder.txt',filemode='w');

import numpy
import matplotlib
from matplotlib import pyplot;
import scipy.ndimage as ndi;

plt = pyplot;
np = numpy;

import pyfits

# ----------------------------------------------------------
# RUN
#
def run(img,thresh_low=0,thresh_high=99,normal=True,seeds=[],truth=[],radius=10,border=0):
    """
    Function to run this segmentation algorithm(s)
    
    Basic input is the image array (float,2D).
    Optionally, a threshold value can be given.
    
    Threshold value is used on bothh segmentation methods: 
     - region growing
     - thresholding
     
    Region Growing is only executed when 'seeds' is given.
    If 'truth' is given, objects matching is done within 'radius'.
    If 'border' is non-null, centroids inside the border are removed.
    
    Input:
     - img   ndarray : Image array (dtype=np.float,ndim=2)
     - thresh  float : Threshold (min) value for segmentation
     - seeds    [()] : Seed point for specific segmentation
     - truth    [()] : Truth table to match with segmented objects
     - radius    int : Distance for tables (objects_vs_truth) matching
     - border    int : Size of the border if/to eliminate objects
    
    Output:
     - prints,logfile and images ...
     
     
    Enjoy.
    ---
    """
    thresh = thresh_low;

    img = img[::-1,:]
    img[img<=0] = 0
    
    logging.info("Image [shape] : (%s)",img.shape)
    logging.debug("Image type : (%s)",img.dtype)

    imin = img.min()
    imax = img.max()
#    img = normalize(img)    # float image
#    logging.info("Image [range] normalized [0:1] : (%.2f,%.2f)",imin,imax)
    
    imgHist_plot = imhist(img)
    imgHist_plot.savefig('img_histogram.png')
    logging.info("Generated a histogram of the image: %s",'img_histogram.png')

    imean = np.mean(img)
    istd = np.std(img)
    logging.info("Image [mean] : (%.2f)",imean)
    logging.info("Image [stdDev] : (%.2f)",istd)


    # Convenience function to save image arrays on PNG files
    def saveImage(img,rootname='imgOut'):
        plt.imshow(img)
        plt.savefig(rootname+'.png')
    
    # And use (MO) binary opening (erosion + dilation) for cleaning spurious regions
    def binary_opening(img_bin):
        strct = ndi.generate_binary_structure(img_bin.ndim,2)
        return ndi.binary_opening(img_bin,strct)


    # Pre-proc
    # --------

    img_clip = histmod(img,thresh_high)   # float image
    logging.info("Removed high-end (>%s) pixels",thresh_high)
    logging.info("Applied [threshold_high] in : %s",img_clip.max())

    fname='out1_imgThreshHighCut'
    saveImage(img_clip,fname)
    pyfits.writeto(fname+'.fits',img_clip,clobber=True)

    if thresh_low == 0:
        thresh_low = histmax(img)
        thresh_low += np.std(img)

    img_clip[img < thresh_low] = 0
    
    img_normal = normalize(img_clip)
    
    fname = 'out2_imgClip'
    saveImage(img_normal,fname)
    pyfits.writeto(fname+'.fits',img_normal,clobber=True)

    img_bin = (img > thresh_low)
    logging.info("Removed low-end (%s<) pixels",thresh_low)    
#    img_bin = binary_opening(img_bin)

    fname = 'out3_imgThreshLowBinOpen'
    saveImage(img_bin,fname)
    pyfits.writeto(fname+'.fits',img_bin.astype(np.int),clobber=True)

    logging.info("Image [binary] generated after binary opening-2 cleaning : (%s)",fname)

#    size = 9
#    img_regions,nlbl = ndi.label(img_bin)
#    for i in xrange(1,nlbl+1):
#        inds = np.where(img_regions==i)
#        if inds[0].size < size:
#            img_regions[inds] = 0
#    logging.info("Removed spurous region (<%d) pixels",size)
#    logging.debug("Number of [regions] last : (%s)",np.unique(img_regions).size)
#    saveImage(img_regions,'out4_imgRegionsTh')
#    
##    img_clean = img * 0
##    img_clean[regions] = img_clip[regions]
##    logging.info("Image [clean] generated with clipped image filtered by regions")
##    saveImage(img_clean,'out5_imgClean')
#
#    # Segment
#    # -------
#    img_smooth = ndi.gaussian_filter(img_normal,3)
#    logging.debug("Image Gauss-smoothed, sigma=%d",3)
#    logging.debug("Image threshold: %.2f",thresh)
#    
#    saveImage(img_smooth,'out6_imgSmooth')
#    
#    if seeds:
#        img_regions = run_rg(img_smooth,seeds,thresh)
#    logging.debug("Segmented IDs: %d",img_regions.max())
#
#    saveImage(img_regions,'out7_imgRegionsRG')
#
#    # Measure
#    # -------
#    img_centroids = center_bright(img_regions,border)
#    idx_centroids = np.where(img_centroids)
#    centroids = zip(idx_centroids[1],idx_centroids[0])
#    logging.debug("Centroids: %s",centroids)
#    
#
#    # Output
#    # ------
#    plt.imshow(img_smooth)
#    plt.savefig('image_smoothed.png')
#    plt.imshow(img_regions)
#    plt.savefig('image_segmented.png')
#    plt.imshow(img_centroids)
#    plt.savefig('image_centroids.png')
#    print "-------------------------------"
#    print "Segmented objects (centroid):"
#    print "Xo Yo"
#    for o_o in centroids:
#        print o_o[0],o_o[1]
#    print "---------"
#
#
#    # Matching
#    # --------
#    if truth:
#        matching_table = match_positions(centroids,truth,radius)
#        matched_elem,neib_x,neib_y = zip(*matching_table)
#        logging.debug("Matched list: %s",matched_elem)
#        logging.debug("Neibors X: %s",neib_x)
#        logging.debug("Neibors Y: %s",neib_y)
#        
#        Ntrue = matched_elem.count(True)
#        Nfalse = matched_elem.count(False)
#        Ntotal_truth = len(truth)
#        Ntotal_sample = len(centroids)
#        print "-------------------------------"
#        print "Matched points (truth neibour):"
#        print "Xo Yo neib? nearest_X nearest_Y"
#        for i in range(len(centroids)):
#            print centroids[i][0],centroids[i][1],matched_elem[i],neib_x[i],neib_y[i]
#        print "-------------------------------"
#        print "Completeness: ",Ntrue/float(Ntotal_truth)
#        print "Contamination: ",Nfalse/float(Ntotal_truth)
#        
#
#    return

# ----------------------------------------------------------
# --- FUNCTIONS --------------------------------------------
# ----------------------------------------------------------

def normalize(img):
    """Normalize 'img'"""
    return (img - img.min())/(img.max() - img.min())
    
def histmax(img):
    """Returns histogram maximum value"""
    imhist,bins = np.histogram(img.flatten(),bins=1000,normed=True)
    return bins[np.argmax(imhist)]

def imhist(img,bins=1000,vmarks=[],log=False,xlabel='X',ylabel='%'):
    """
    Plot histogram from image array
    #TODO This function should not remain here.
    
    Output:
     - plt  : pyplot object. plt.show() will give you the histogram.
    
    """
    np = numpy
    
    hist,bins = np.histogram(img.flatten(),bins=bins,normed=True)
    
    return barhist(hist,bins,vmarks,log,xlabel,ylabel)

def barhist(hist,bins,vmarks=[],log=False,xlabel='X',ylabel='%'):
    """
    Plot histogram with vertical line marks
    
    Output:
     - plt  : pyplot object. plt.show() will give you the histogram.
    
    """
    plt = matplotlib.pyplot;
        
    width = 0.7*(bins[1]-bins[0])
    centers = (bins[:-1]+bins[1:])/2.
    plt.grid(True)
    plt.bar(centers, hist, width=width, align='center', color='g', log=log, alpha=0.9)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    ymax = hist.max()*1.01
    ymin = -ymax/50.
    
    for i,iv in enumerate(vmarks):
        ymaxn = ymax + i%2*(ymin*2)
        plt.vlines(iv,ymin,ymaxn,color='r')
        plt.text(iv,ymaxn,"%.1f"%(iv))
        
    return plt
#
# === FILTERING ===
#

def histmod(img,val=99,nbins=1000):
    """
    Cuts off the highest (intensity) pixels of image
    
    Input:
     - img ndarray : Image array
     - val     int : Volume (percent) image pixels to maintain [0:100]
     - nbins   int : Number of bins to use for equalization
    
    Output:
     - img_clip  ndarray : Clipped (at 90%) image array
    
    ---
    """
    
    imhist,bins = np.histogram(img.flatten(),nbins,normed=True);
    
    cdf = np.cumsum(imhist);
    cdf = cdf/cdf[-1];
    ind = list(cdf > val/100.).index(1)
    cut_min = img.min()
    cut_max = bins[ind]
    
    return np.clip(img,cut_min,cut_max)

#
# === SEGMENTATION ===
#

def thresholding(img,thresh,size=9):
    """
    Segment using a thresholding algorithm
    
    Input:
     - img  ndarray : Image array (ndim=2)
     - thresh float : Threshold value for pixels selectino
     - size     int : Minimum size a group of pixels must have
    
    Output:
     - regions : Binary array for each segmented region
    
    ---
    """

    logging.debug("Threshold: %.2f",thresh)
    logging.debug("Objects min size: %d",size)
    
    # Take the binary image thresholded
    img_bin = (img > thresh)
    
    # And use (MO) binary opening (erosion + dilation) for cleaning spurious Trues
    strct = ndi.generate_binary_structure(2,2)
    img_bin = ndi.binary_opening(img_bin,strct)
    
    # Label each group/region (value==True) of pixels
    regions,nlbl = ndi.label(img_bin)
    for i in xrange(1,nlbl+1):
        inds = np.where(regions==i)
        if inds[0].size < size:
            regions[inds] = 0

    logging.debug("Threshold labels: %s",np.unique(regions))

    return regions.astype(np.bool)


def region_growing(img,x,y,thresh):
    """
    Segment using a Region Growing algorithm
    
    Input:
     - img  ndarray : Image array (ndim=2)
     - x        int : Seed x position
     - y        int : Seed y position
     - thresh float : Threshold value for limiting the grow
    
    Output:
     - region  : Region grown around given 'x,y'
    
    ---
    """

    x_o = x
    y_o = y
    flag = True

    logging.debug("RG seed: X(%d) Y(%d)",x_o,y_o)
    logging.debug("Threshold: %.2f",thresh)

    # Initialize region with the seed point
    region = np.zeros(img.shape,dtype=np.bool)
    reg_old = (region==flag)

    logging.debug("Image seed point value: %.2f",img[y_o,x_o])
    if (img[y_o,x_o] < thresh): return region
    
    region[y_o,x_o] = flag
    reg_cur = (region==flag)

    # For future (loop) morphological operations...
    strct_elem = ndi.generate_binary_structure(2,2)

    # While region stills changes (grow), do...
    while not np.all(region == reg_old):
        
        reg_old = region.copy()

        # Define pixel neighbours
        tmp = ndi.binary_dilation(region,strct_elem, 1)
        neigbors = tmp - region
        inds = np.where(neigbors)

        # Check for the new neighbors; do they fullfil requirements?
        #region[np.where(region[inds]>=thresh)] = flag
        for y_i,x_i in zip(*inds):
            if (img[y_i,x_i] >= thresh):
                region[y_i,x_i] = flag

    return region

#
# === MEASURES ===
#

def center_bright(img,border=0):
    """
    Returns an array with the image/region centroids
    
    Input:
     - img     ndarray : Image array (dtype=np.uint,ndim=2)
     - border      int : Size of border if want to remove near-border objects
                         Default is '0' for *no* border-objects removal
    
    Output:
     - seeds : Same 'img' size array with labeled seed points

    ---
    """
    
    regs = img.copy()

    logging.debug("Border size: %d",border)
    
    # Remove maxima found near borders
    if border:
        brd = int(border)
        border = np.zeros(regs.shape,np.bool)
        border[-brd:,:]=border[:brd,:]=border[:,-brd:]=border[:,:brd] = 1
        for id in np.unique(regs[border]):
            regs[regs==id] = 0

    # Read out the centroids (x_o,y_o points)
    cms = np.zeros(img.shape,np.uint)
    for id in np.unique(regs[regs>0]):
        cms_tmp = cms*0
        cms_tmp[regs==id] = 1
        ym,xm = ndi.center_of_mass(cms_tmp)
        cms[ym,xm] = id
        logging.debug("Center of brightness: X(%d) Y(%d)",xm,ym)
    
    return cms

#
# === MATCHING ===
#

def nearest_neighbour(centroids_A, centroids_B):
    """
    Function to compute and return the set of nearest point
        
    This function computes for each entry of 'centroids_A', the
    nearest point in 'centroids_B'.
    
    Is returned, respective to each entry, the index and the
    distance measured correspondig to each identified nearest
    point.
    
    Input:
     - centroids_A : [(x_A0,y_A0),(x_A1,y_A1), ]
     - centroids_B : [(x_B0,y_B0),(x_B1,y_B1), ]
    
    Output:
     - [(index_BtoA0,distance_BtoA0), ]

    ---
    """
    
    length_A = len(centroids_A);
    length_B = len(centroids_B);
    x_A,y_A = zip(*centroids_A);
    x_B,y_B = zip(*centroids_B);
    
    Mat = np.zeros((length_A,length_B));
    for i in xrange(length_A):
        Mat[i] = (x_A[i] - x_B[:])**2 + (y_A[i] - y_B[:])**2;
    
    dist_min_BtoA = np.sqrt(np.amin(Mat,axis=1));
    indx_min_BtoA = np.argmin(Mat,axis=1);
    
    return zip(indx_min_BtoA,dist_min_BtoA)


def match_positions(centroids,truth,radius):
    """
    Check if positions in tables match within a given radius
    
    Input:
     - centroids  [()] : list of (x,y) points
     - truth      [()] : list of (x,y) points
     - radius    float : Distance parameters for objects matching
    
    Output:
     - matched_CentTruth  : List (lenght == centroids) with "triples"
                            "triplets" elements are:
                             - Is there a near neibour? True|False
                             - The nearest truth x point? integer
                             - The nearest truth y point? integer
    
    ---
    """
        
    Cent_near_neibors = nearest_neighbour(centroids,truth)
    Cent_nn_indx,Cent_nn_dist = zip(*Cent_near_neibors)
    
    Cent_match_neibors = [ float(_d) < float(radius) for _d in Cent_nn_dist ]
    Cent_x,Cent_y = zip(*centroids)
    
    Cent_neibor_Tx = [ truth[i][0] for i in Cent_nn_indx ]
    Cent_neibor_Ty = [ truth[i][1] for i in Cent_nn_indx ]
    
    Cent_Truth_matched = zip(Cent_x,Cent_y,Cent_match_neibors,Cent_neibor_Tx,Cent_neibor_Ty)
    
    return Cent_Truth_matched


# --------------------------------------------

#
# === RUN SEGMENTERS LOOP ===
#

def run_thresh(img,seeds,thresh,size=9):
    """
    Segment using a thresholding algorithm
    
    Input:
     - img  ndarray : Image array (ndim=2)
     - seeds   [()] : List with (x,y) seeds: [(x0,y0),(x1,y1),...]
     - thresh float : Threshold value for pixels selectino
     - size     int : Minimum size a group of pixels must have
    
    Output:
     - regions : Binary array for each segmented region
    
    ---
    """

    img_bin = thresholding(img,thresh,size)
    img_labeled,nlbl = ndi.label(img_bin)

    regions = np.zeros(img.shape,np.int)
    id = 0
    for xy in seeds:
        x_o,y_o = xy
        seed_lbl = img_labeled[y_o,x_o]
        if seed_lbl == 0: continue
        id += 1
        regions[img_labeled==seed_lbl] = id

    return regions


def run_rg(img,seeds,thresh):
    """
    Segment using a Region Growing algorithm
    
    Input:
     - img  ndarray : Image array (ndim=2)
     - seeds   [()] : List with (x,y) seeds: [(x0,y0),(x1,y1),...]
     - thresh float : Threshold value for limiting the grow
    
    Output:
     - regions  : Image array with segmented (labeled) regions
    
    ---
    """

    regions = np.zeros(img.shape,np.int)
    
    for id,xy in enumerate(seeds,1):
        x_o,y_o = xy
        reg_true = region_growing(img,x_o,y_o,thresh)
        regions[reg_true] = id

    return regions

# --------------------------------------------

if __name__ == '__main__':
    
    print run.__doc__
    print dir(run)
    
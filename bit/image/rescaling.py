import numpy as np;
from image import Distros

# ---
def tanh(img,slope=100,cut=None):
    """
    Modifies image intensity by an hyperbolic tangent function
    """
    
    new_img = np.zeros(img.shape,dtype=img.dtype);
    
    min = img.min();
    max = img.max();
    
    if cut == None:
        steps = 10000;
        delta_h = float(max-min)/(steps-1);
        hist,bins = np.histogram(img.flatten(),bins=steps);
        cut = min + (hist.argmax()+1) * delta_h;
        print "Hist max:",cut
    
    new_img = np.tanh(slope*(img-cut)) + 1.;
    
    return new_img;


def equalization(img,nbins=1000):
    """
    Equalizes image histogram
    
    Input:
     - img ndarray : Image array
     - nbins int : Number of bins to use for equalization
    
    Output:
     - img_eq  ndarray : Image array equalized
    """
    
    cdf,bins = Distros.cdf(img,nbins)
    imgterp = np.interp(img.flatten(),bins[:-1],cdf)
    img_eq = imgterp.reshape(img.shape)
    return img_eq;

def clip(img,thresholds=[None,None],fill=[None,None]):
    """
    Clip image intensities, below and above thresholds

    Arguments passed valueing 'None' will not be modified. For example,
    if "thresholds[0]=None", the minimum value will remain untouched.

    The 'fill' values, if given, will be used to substitute the values 
    below (not-equal) and/or above (not-equal) to the respective thresholds.
    """

    cut_min,cut_max = thresholds
    if cut_min is None and cut_max is None:
        return img
    if cut_min is None:
        cut_min = img.min()
    if cut_max is None:
        cut_max = img.max()
    if cut_min > cut_max:
        tmp = cut_max
        cut_max = cut_min
        cut_min = tmp
    if cut_min==cut_max:
        return img*np.nan

    fill_min,fill_max = fill
    indx_min = None
    indx_max = None
    if fill_min!=None or fill_max!=None:
        if fill_min!=None:
            indx_min = np.where(img<cut_min)
        if fill_max!=None:
            indx_max = np.where(img>cut_max)

    imclip = np.clip(img,cut_min,cut_max)
    if indx_min!=None:
        imclip[indx_min] = fill_min
    if indx_max!=None:
        imclip[indx_max] = fill_max
    return imclip


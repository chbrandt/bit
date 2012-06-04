import logging
from scipy import weave;
import numpy as np;

# ---
def tanh(img,slope=100,cut=None):
    """
    Modifies image intensities scale
    
    Input:
     - img <ndarray>
     - slope <float> : tanh('slope'*image)
     - cut <float> : If 'cut' is not given, histogram's maximum is used
    
    Output:
     - ndarray
    
    ---
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

# ---
def histo_equal(img,nbins=1000,savefig=True,figname='hist_equal.png'):
    """
    Equalizes given image histogram
    
    Input:
     - img ndarray : Image array
     - nbins int : Number of bins to use for equalization
    
    Output:
     - img_eq  ndarray : Image array equalized
    
    ---
    """
    
    imhist,bins = np.histogram(img.flatten(),nbins,normed=True)
    cdf = np.cumsum(imhist)
    cdf = cdf/cdf[-1]
    imgterp = np.interp(img.flatten(),bins[:-1],cdf)
    img_eq = imgterp.reshape(img.shape)
    
    if savefig:
        eqhist,bins = np.histogram(img_eq.flatten(),nbins,normed=True);
        cdfout = np.cumsum(eqhist);
        cdfout = cdfout/cdfout[-1];
        figout = plotfig(img,imhist,cdf,bins,img_eq,eqhist,cdfout);
        figout.savefig(figname);
        
    return img_eq;
    
# ---
def max_percent(img,val=90,nbins=1000,savefig=True,figname='hist_equal.png'):
    """
    Cuts off the edges of given image histogram
    
    Input:
     - img ndarray : Image array
     - val int : [0:100]; '0' removes the lower values than fashion (histogram)
     - nbins int : Number of bins to use for equalization
    
    Output:
     - img_eq  ndarray : Image array equalized
    
    ---
    """
    
    imhist,bins = np.histogram(img.flatten(),nbins,normed=True);
    logging.debug("Hist: %s",imhist)
    logging.debug("Bins: %s",bins)
    
    if val:
        logging.debug("percent cut: computing pixels volume for %d\% cut",val)
        cdf = np.cumsum(imhist);
        cdf = cdf/cdf[-1];
        ind = list(cdf > val/100.).index(1)
        cut_min = img.min()
        cut_max = bins[ind]
        logging.debug("index where volume cuts: %d, min: %.2f, max: %.2f",ind,cut_min,cut_max)
    
    else:
        logging.info("auto threshold: using distribution's fashion value")
        ind = np.argmax(imhist)
        cut_min = bins[ind]
        cut_max = img.max()
        logging.info("histogram argmax: %d, min: %.2f, max: %.2f"%(ind,cut_min,cut_max))
    
    img_clipd = np.clip(img,cut_min,cut_max)
    
    return img_clipd;

import logging

import numpy as np


def equal(img,nbins=1000,savefig=True,figname='hist_equal.png'):
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
    

def thresh(img,val=90,nbins=1000,savefig=True,figname='hist_equal.png'):
    """
    Cuts off the edges of given image histogram
    
    Input:
     - img ndarray : Image array
     - val int : [0:100]; '0' removes the background at the fashion value
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

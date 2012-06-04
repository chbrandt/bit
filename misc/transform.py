import numpy as np

# -
def histeq(img,nbins=10000):
    """
    Equalizes given image histogram
    
    Input:
     - img ndarray : Image array
     - nbins int : Number of bins to use for equalization
    
    Output:
     - img_eq  ndarray : Image array equalized
    
    ---
    """
    
    imhist,bins = np.histogram(img.flatten(),nbins,normed=True);
    
    cdf = np.cumsum(imhist);
    cdf = cdf/cdf[-1];
    
    imgterp = np.interp(img.flatten(),bins[:-1],cdf);
    
    img_eq = imgterp.reshape(img.shape);
    
    return img_eq;

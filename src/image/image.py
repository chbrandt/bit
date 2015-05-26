"""Basic image/array processing, transforms and analysis."""

import logging
import numpy
np = numpy
from scipy import ndimage
try:
    import pymorph
except:
    pass

class Distros:
    @staticmethod
    def histogram(img,nbins=1000,normed=True):
        imhist,bins = np.histogram(img.flatten(),nbins=nbins,normed=normed)
        return imhist,bins

    @staticmethod
    def cdf(img,nbins):
        """
        The Cumulative Distribution Function
        """
        hist,bins = Distros.histogram(img,nbins)
        imcdf = np.cumsum(hist);
        imcdf/=imcdf[-1];
        return imcdf,bins


class Stats:
    @staticmethod
    def percentValue(img,val=0.9)
        cdf = Distros.cdf(img)
        val = val if val<1 and val>0 else abs(val/100.)
        ind = list(cdf > val).index(1)
        val = bins[ind]
        return val

    @staticmethod
    def mode(img):
        hist,bins = Distros.histogram(img)
        ind = np.argmax(imhist)
        val = bins[ind]
        return val


# ---
def invert(img,max=None):
    """
    Returns de (grayscale) inverse image
    """
    np = numpy;
    
    max_val = img.max()
    img_inv = np.abs(max_val-img)
    
    return img_inv

# ---
def normalize(img,unit=1):
    """
    Normalize 'img' to 'unit'
    """
    
    return unit * (img - img.min())/(img.max() - img.min())
    
# ---
def float2uint(img):
    """
    Normalize and truncate 'img' values to uint8 scale [0:255]
    """
    np = numpy;
    
    img = normalize(img) * 255.
    u_img = img.astype(np.uint8)
    
    return u_img

# ---
def grad(img):
    """Returns image gradiente (module^2) array"""
    np = numpy;
    
    gx,gy = np.gradient(img)
    grad_img = gx**2 + gy**2

    return grad_img

# ---
def grid(X=(-5,5,0.1),Y=(-5,5,0.1)):
    """
    Returns the pair X,Y of grid coordinates
    
    Input:
     - shape  (int,int) : Sizes of X,Y sides
     - res        float : point resolution
    
    Output:
     - X,Y  ndarrays : x,y meshgrids
    
    ---
    """
    
    X = list(X)
    Y = list(Y)
    X[1] += X[2]
    Y[1] += Y[2]
    X = np.arange(*X)
    Y = np.arange(*Y)
    X,Y = np.meshgrid(X,Y)

    return X,Y
    
# ---
def combine(backimg, frontimg, x, y):
    """
    Add (merge) two given images centered at (x,y)
    
    First argument, 'groundimg', is used as base array for the merging
    process, where 'topimg' will be added to. 'x' and 'y' are the
    coordinates (on 'grounimg') where 'topimg' central point will be
    placed.
    
    Note/Restriction: groundimg.shape >= topimg.shape
    
    Input:
     - groundimg : numpy.ndarray(ndim=2)
     - topimg : numpy.ndarray(ndim=2)
     - x : int
     - y : int
    
    Output:     
     - merged image : numpy.ndarray(ndim=2)
     
     ---
     """

    groundimg = backimg
    topimg = frontimg
    
    if (groundimg.shape != topimg.shape):
        return False;

    x = int(x);
    y = int(y);
    
    logging.debug('Reference position for adding images: (%d, %d)' % (x,y))

    DY,DX = topimg.shape;
    y_img_size,x_img_size = groundimg.shape;
    
    DY2 = int(DY/2);
    if ( DY%2 ):
        y_fin = y+DY2+1;
    else:
        y_fin = y+DY2;
    y_ini = y-DY2;

    DX2 = int(DX/2);
    if ( DX%2 ):
        x_fin = x+DX2+1;
    else:
        x_fin = x+DX2;
    x_ini = x-DX2;
    
    # Define the images (in/out) slices to be copied..
    #
    x_ini_grd = max( 0, x_ini );   x_fin_grd = min( x_img_size, x_fin );
    y_ini_grd = max( 0, y_ini );   y_fin_grd = min( y_img_size, y_fin );

    x_ini_top = abs( min( 0, x_ini ));   x_fin_top = DX - (x_fin - x_fin_grd);
    y_ini_top = abs( min( 0, y_ini ));   y_fin_top = DY - (y_fin - y_fin_grd);

    groundimg[y_ini_grd:y_fin_grd,x_ini_grd:x_fin_grd] += topimg[y_ini_top:y_fin_top,x_ini_top:x_fin_top];

    return groundimg;
import numpy as np

# --

def grid(X=(-5,5,0.1),Y=(-5,5,0.1)):
    """
    Returns the pair X,Y of grid coordinates
    
    Input:
     - shape  (int,int) : Sizes of X,Y sides
     - res        float : point resolution
    
    Output:
     - X,Y  ndarrays : x,y meshgrids
    
    ---
    """
    
    X = list(X)
    Y = list(Y)
    X[1] += X[2]
    Y[1] += Y[2]
    X = np.arange(*X)
    Y = np.arange(*Y)
    X,Y = np.meshgrid(X,Y)

    return X,Y
    
# --

def sersic(R=0,n=0.5,b=1,X=(-5,5,0.1),Y=(-5,5,0.1)):
    """
    Returns an intensity array of a Sersic profile
    
    Input:
    
    Output:
    
    ---
    """
    
    X,Y = grid(X,Y)

    if not R:
        R = np.sqrt(X**2 + Y**2)
        R = 2*R/R.max()
    else:
        R = np.sqrt(X**2 + Y**2)/R

    Z = np.exp( -b * (np.power(R,1./n) - 1))

    return Z


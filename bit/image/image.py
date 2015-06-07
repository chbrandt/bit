"""Basic image/array processing, transforms and analysis."""

import logging
import numpy as np
from scipy import ndimage as ndi

try:
    import pymorph
    _PYMORPH=True
except:
    _PYMORPH=False

class Distros:
    @staticmethod
    def histogram(img,nbins=1000,normed=True):
        """
        Return image's histogram
        """
        imhist,bins = np.histogram(img.flatten(),nbins=nbins,normed=normed)
        return imhist,bins

    @staticmethod
    def cdf(img,nbins=1000):
        """
        Return the Cumulative Distribution Function of given image
        """
        hist,bins = Distros.histogram(img,nbins)
        imcdf = np.cumsum(hist);
        imcdf/=imcdf[-1];
        return imcdf,bins


class Stats:
    @staticmethod
    def cdv(img,percent=0.9):
        """
        Return image's intensity where CDF sum to 'percent' value.
        """
        _cdf,bins = Distros.cdf(img)
        perc = percent if 0<percent<1 else abs(percent/100.)
        ind = list(_cdf > perc).index(1)
        _cdv = bins[ind]
        return _cdv

    @staticmethod
    def quartiles(img):
        """
        Return image's intensity value related to CDF's 'perc' value.
        """
        _cdf,bins = Distros.cdf(img)
        vals = []
        for pq in [0.25,0.5,0.75]:
            ind = list(_cdf > pq).index(1)
            vals.append(bins[ind])
        assert(len(vals)==3)
        return vals

    @staticmethod
    def mode(img):
        """
        Return image's mode value
        """
        _hist,bins = Distros.histogram(img)
        ind = np.argmax(_hist)
        _mode = bins[ind]
        return _mode



class Transf:
    # ---
    @staticmethod
    def invert(img,max=None):
        """
        Return the inverse(negative) of image
        """
        _max = img.max()
        _min = img.min()
        img_inv = (_max - img) + _min
        return img_inv
    
    # ---
    @staticmethod
    def normalize(img,unit=1):
        """
        Normalize image intensity range
        """
        _min = img.min()
        _rng = img.max() - _min
        _fc = float(unit)/_rng
        img_norm = img - _min
        return img_norm * _fc 
        
    # ---
    @staticmethod
    def float2uint(img):
        """
        Normalize image to uint8 scale range [0:255]
        """
        img = Transf.normalize(img,255)
        u_img = img.astype(np.uint8)
        return u_img


class Profile:
    # ---
    def grad(img):
        """
        Return image gradiente (module^2) array
        """
        gx,gy = np.gradient(img)
        grad_img = gx**2 + gy**2
        return grad_img
    
    # ---
    def grid(X=(-5,5,0.1),Y=(-5,5,0.1)):
        """
        Return the pair X,Y of grid coordinates
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
    def sersic(R=0,n=0.5,b=1,X=(-5,5,0.1),Y=(-5,5,0.1)):
        """
        Return an image of the Sersic profile defined by R,n,b
        """
        X,Y = Profile.grid(X,Y)
        if not R:
            R = np.sqrt(X**2 + Y**2)
            R = 2*R/R.max()
        else:
            R = np.sqrt(X**2 + Y**2)/R
        Z = np.exp( -b * (np.power(R,1./n) - 1))
        return Z

# ---
#def combine(backimg, frontimg, x, y):
#    """
#    Add (merge) two given images centered at (x,y)
#    
#    First argument, 'groundimg', is used as base array for the merging
#    process, where 'topimg' will be added to. 'x' and 'y' are the
#    coordinates (on 'grounimg') where 'topimg' central point will be
#    placed.
#    
#    Note/Restriction: groundimg.shape >= topimg.shape
#    
#    Input:
#     - groundimg : numpy.ndarray(ndim=2)
#     - topimg : numpy.ndarray(ndim=2)
#     - x : int
#     - y : int
#    
#    Output:     
#     - merged image : numpy.ndarray(ndim=2)
#     
#     ---
#     """
#
#    groundimg = backimg
#    topimg = frontimg
#    
#    if (groundimg.shape != topimg.shape):
#        return False;
#
#    x = int(x);
#    y = int(y);
#    
#    logging.debug('Reference position for adding images: (%d, %d)' % (x,y))
#
#    DY,DX = topimg.shape;
#    y_img_size,x_img_size = groundimg.shape;
#    
#    DY2 = int(DY/2);
#    if ( DY%2 ):
#        y_fin = y+DY2+1;
#    else:
#        y_fin = y+DY2;
#    y_ini = y-DY2;
#
#    DX2 = int(DX/2);
#    if ( DX%2 ):
#        x_fin = x+DX2+1;
#    else:
#        x_fin = x+DX2;
#    x_ini = x-DX2;
#    
#    # Define the images (in/out) slices to be copied..
#    #
#    x_ini_grd = max( 0, x_ini );   x_fin_grd = min( x_img_size, x_fin );
#    y_ini_grd = max( 0, y_ini );   y_fin_grd = min( y_img_size, y_fin );
#
#    x_ini_top = abs( min( 0, x_ini ));   x_fin_top = DX - (x_fin - x_fin_grd);
#    y_ini_top = abs( min( 0, y_ini ));   y_fin_top = DY - (y_fin - y_fin_grd);
#
#    groundimg[y_ini_grd:y_fin_grd,x_ini_grd:x_fin_grd] += topimg[y_ini_top:y_fin_top,x_ini_top:x_fin_top];
#
#    return groundimg;


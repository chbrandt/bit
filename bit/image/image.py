# -*- coding:utf-8 -*-

"""
Basic image processing functions.
"""


def normalize(img,unit=1):
    """
    Normalize image intensity range. Default is to [0:1]
    """
    _min = img.min()
    _rng = img.max() - _min
    _fc = float(unit)/_rng
    img_norm = img - _min
    return img_norm * _fc 
    
def float2uint(img):
    """
    Normalize image to uint8 scale range [0:255]
    """
    img = normalize(img,255)
    u_img = img.astype(np.uint8)
    return u_img

def invert(img,max=None):
    """
    Invert image intensity values (i.e, returns the negative image)
    """
    _max = img.max()
    _min = img.min()
    img_inv = (_max - img) + _min
    return img_inv


import numpy as np


class Distros:
    """
    Namespace to group functions computing/returning distributions
    """
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
    """
    Namespace to group functions computing statistical values
    """
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
    """
    Namespace to group functions for basic iamge values transformations.
    """
    @staticmethod
    def grad(img):
        """
        Return image gradiente (module^2) array
        """
        gx,gy = np.gradient(img)
        grad_img = gx**2 + gy**2
        return grad_img
    

class Profile:
    """
    Namespace to group functions computing/returning distributions
    """
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


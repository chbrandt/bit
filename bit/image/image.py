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


def invert(img):
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
        imhist,bins = np.histogram(img.flatten(),bins=nbins,normed=normed)
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
    Namespace for methods dealing with
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


    def maxima(img):
        """
        Returns image local maxima
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


class Model:
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

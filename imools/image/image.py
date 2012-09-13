#
# Carlos Brandt - carloshenriquebrandt@gmail.com
# 4.* Imools - Out/2011
#
"""Basic image/array processing, transforms and analysis."""

import numpy
from scipy import ndimage
import pymorph


# ---

def invert(img,max=None):
    """
    Returns de (grayscale) inverse image
    """
    np = numpy;
    
    max_val = img.max()
    img_inv = np.abs(max_val-img)
    
    return img_inv

# --

def normalize(img):
    """
    Normalize 'img'
    """
    
    return (img - img.min())/(img.max() - img.min())
    
# --

def float2uint(img):
    """
    Normalize and truncate 'img' values to uint8 scale [0:255]
    """
    np = numpy;
    
    img = normalize(img) * 255
    u_img = img.astype(np.uint8)
    
    return u_img

# --

def grad(img):
    """Returns image gradiente (module^2) array"""
    np = numpy;
    
    gx,gy = np.gradient(img)
    grad_img = gx**2 + gy**2

    return grad_img

# --

def local_maxima(img):
    """
    Returns image local maxima
    
    Input:
     - img  ndarray : Image array (ndim=2,dtype=float|int)
    
    Output:
     - maxs : Same 'img' size array with maxima = True (dtype=bool)
    
    ---
    """
    ndi = ndimage;

    # Image has to be 'int' [0:255] to use in 'pymorph'
    img = float2uint(img)

    # Search for image maxima
    maxs = pymorph.regmax(img)

    # A closing step is used "clue" near maxima
    elem_strct = ndi.generate_binary_structure(2,2)
    maxs = ndi.binary_closing(maxs,elem_strct)

    return maxs

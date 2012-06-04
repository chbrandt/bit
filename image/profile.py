import numpy as np
import image
from scipy import ndimage as ndi


# ---
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

# ---
def gauss_kernel(sigma,size=5):
    """ 
    Creates a normalized 2D isotropic gaussian kernel array for convolution.
    The size of the 2D gaussian kernel array is defined as a multiple (n_fwhm)
    of the FWHM.

    Input:
     - sigma  <int> : sigma (StdDev) of the gaussian kernel
     - size   <int> : Defines the size of the kernel window

    Output:
    - <ndarray>: normalized 2D gaussian kernel array for convolution
    """
    np = numpy;
    
    sigma = float(sigma)
    sigma_x = sigma_y = sigma
    
    x_sz = y_sz = int(size * sigma + 0.5)
    x, y = np.mgrid[-x_sz/2:x_sz/2,-y_sz/2:y_sz/2]
    
    g_kern = np.exp(-(x**2/(2*(sigma_x**2))+y**2/(2*(sigma_y**2))))
    
    return g_kern / g_kern.sum()

# ---
def moffat_kernel(beta,radius,size=4):
    """
    Creates a normalized 2D Moffat kernel array for convolution.
    The size of the 2D kernel array is defined as a multiple (n_fwhm)
    of the FWHM.
     
    Input:
    - beta <float>: profile slope
    - radius <float>: scale radius
    - size <int>: Defines the size of the kernel window

    Output:
    - <ndarray>: normalized 2D kernel array for convolution
    """
    np = numpy;
    
    x_sz = y_sz = int(size * sigma + 0.5)
    x, y = np.mgrid[-x_sz/2:x_sz/2,-y_sz/2:y_sz/2]

    m_kern = 1. / ((1+(x**2+y**2)/radius**2)**beta)
    
    return m_kern / m_kern.sum()
    
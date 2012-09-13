import numpy
import scipy.ndimage as ndi

# ---

def average(img, size=3):
    """
    Simple mean value filter

    Input:
     - img <ndarray>
     - size <int> : size of the filter window

    Output:
     <ndarray>
    
    ---
    """
    np = numpy
    
    size = int(size);
    kernel = np.ones((size,size)) / float(size**2);
    
    return ndi.convolve(img,kernel)

mean = average

# --

def median(img, size=3):
    """
    Simple median value filter
    
    Input:
     - img <ndarray>
     - size <int> : size of the filter window
    
    Output:
     <ndarray>
    
    ---
    """
    
    return ndi.median_filter(img,size);

# --

def gaussian(img, sigma=[3,3]):
    """
    Simple gaussian filter
    
    Input:
     - img <ndarray>
     - sigma <[int,int]> : sigma window
    
    Output:
     - <ndarray>
    
    ---
    """

    return ndi.gaussian_filter(img,sigma);

# --

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

# --

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
    
# --

def directional(img, size=3, get_val='min'):
    """
    Directional smooth filter
    
    Input:
     - img <ndarray>
     - size <int> : array length to apply the filter
     - get_val <str> : it controls whether the value computed is the minimum or maximum
    
    Output:
     - ndarray
    
    ---
    """
    np = numpy;
    
    y_size, x_size = img.shape;
    
    new_img = img.copy();
    
    hsize = int(size/2);
    indx_vec = np.arange(-hsize,hsize+1);
    
    for lin in xrange(hsize,y_size-hsize):
        for col in xrange(hsize,x_size-hsize):
        
            sum_d1, sum_d2, sum_h, sum_v = 0.,0.,0.,0.;
            for i in indx_vec:
                sum_d1 += img[lin+i,col+i];
                sum_d2 += img[lin-i,col+i];
                sum_h += img[lin,col+i];
                sum_v += img[lin+i,col];
            
            dif_d1 = abs((sum_d1/size) - img[lin,col]);
            dif_d2 = abs((sum_d2/size) - img[lin,col]);
            dif_h = abs((sum_h/size) - img[lin,col]);
            dif_v = abs((sum_v/size) - img[lin,col]);
            
            if get_val == 'min':
                valor = min(dif_d1,dif_d2,dif_h,dif_v) + img[lin,col];
            else:
                valor = max(dif_d1,dif_d2,dif_h,dif_v) + img[lin,col];

            new_img[lin,col] = valor;
    
    return new_img;
    

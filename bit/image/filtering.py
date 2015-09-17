import numpy as np
import scipy.ndimage as ndi


def mean(img, size=(3,3)):
    """
    Mean filter

    The mean filter evaluates the mean (intensity) values among the surrounding
    pixels and substitute the central pixel value by the just computed mean.
    Neighbor pixels (i.e, "surrounding") are defined by the 'size' argument, where
    the central pixel is (relatively) located at (size/2,size/2).

    This function is simply a common/daily interface for scipy ndimage's convolve,
    where the 'weights' array is a all-constant array given by 1/size.

    Arguments:
    - img : ndarray (int|float)
        Numpy array representing the image/pixel values
    - size : tuple (int)
        (y,x) sizes for the filtering window

    Returns:
    - ndarray (float)
        Numpy array representing the filtered image
    """
    # sanity check: guarantee the kernel window to have integer size values!
    y_size = int(size[0])
    x_size = int(size[1])
    kernel = np.ones((y_size,x_size)) / float(y_size*x_size);
    return ndi.convolve(img,kernel)


def median(img, size=(3,3)):
    """
    Median filter

    The median filter evaluates the median (intensity) value among the surrounding
    pixels and substitute the central pixel value by the just computed median.
    Neighbor pixels (i.e, "surrounding") are defined by the 'size' argument, where
    the central pixel is (relatively) located at (size/2,size/2).

    This function is simply an alias to scipy ndimage's median_filter.

    Arguments:
    - img : ndarray (int|float)
        Numpy array representing the image/pixel values
    - size : tuple (int)
        (y,x) sizes for the filtering window

    Returns:
    - ndarray (float)
        Numpy array representing the filtered image
    """
    return ndi.median_filter(img,size);


def gaussian(img, sigma=[3,3]):
    """
    Gaussian filter

    Alias to Scipy ndimage's gaussina_filter function.

    Arguments:
    - img : ndarray (int|float)
        Numpy array representing the image/pixel values
    - sigma : tuple (int)
        (y,x) sizes for the filtering window, the gaussian standard deviation

    Returns:
    - ndarray (float)
        Numpy array representing the filtered image
    """
    return ndi.gaussian_filter(img,sigma);


def directional(img, size=3, get_val='min'):
    """
    Directional smooth filter

    The filter evaluates the mean value across the 4 main directions crossing
    each pixel (primary and secondary diagonals, vertical and horizontal), and
    based on the 'get_val' argument defines which value to use as substitution
    for the central pixel. 'size' relates to the length of each directional
    vector to compute the mean.

    This filter is a cheap implementation of a gradient-based filter. The directional
    decision assumes a certain kind of simetry on the image intensity profile;
    which for the case of astronomical images is pretty reasonable.
    By all means, qualitatively (and roughly) speaking, the choice for 'get_val=min'
    should output a more smooth (delicate) image; the choice for 'get_val=max'
    should enhance the image contrast.

    Arguments:
    - img : ndarray (int|float)
        Numpy array representing the image/pixel values
    - size : int
        Size/length for the filtering vector (in each direction)
    - get_val : 'min'|'max'
        Use 'min' for the mean value of the least intensity varying direction,
        'max' for the most (intensity) verying direction.

    Returns:
    - ndarray (float)
        Numpy array representing the filtered image
    """
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

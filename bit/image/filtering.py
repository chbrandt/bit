import numpy as np
import scipy.ndimage as ndi


def mean(img, size=(3,3)):
    """
    Simple mean value filter
    """
    kernel = np.ones((int(size[0]),int(size[1]))) / float(size[0]*size[1]);
    return ndi.convolve(img,kernel)


def median(img, size=(3,3)):
    """
    Simple median value filter
    """
    return ndi.median_filter(img,size);


def gaussian(img, sigma=[3,3]):
    """
    Simple gaussian filter
    """
    return ndi.gaussian_filter(img,sigma);


def directional(img, size=3, get_val='min'):
    """
    Directional smooth filter
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
    
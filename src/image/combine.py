"""Combine two or more images"""

##@package combine

import string
import logging

def overlap(groundimg, topimg, x, y):
    """
    Sum two image arrays, respecting position (x,y) given
    
    This is a simple summing of arrays, the thing about this function is
    the alignment it does for the addition. '(x,y)' should be the point
    (coordinate) at 'groundimg' to where 'topimg' will be center-aligned.

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

    if groundimg.shape[0] < topimg.shape[0]  or  groundimg.shape[1] < topimg.shape[1]:
        #TODO: this is a very weak restriction. Should be removed!
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

    return (groundimg);


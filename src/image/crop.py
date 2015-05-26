"""Package to make FITS image copy/cuts and update their header"""

import numpy as np;

# ---
def cutout(image,xo=0,yo=0,x_size=0,y_size=0,mask=None):
    """
    Make a cutout from image array

    If 'xo=0' and 'yo=0', given image (input_file) central pixel will be chosen as (xo,yo).
    If 'x_size=0' and 'y_size=0', half length of each side will be used for output dimensions.

    Input:
     - image           :ndarray: Image array (ndim=2,dtype=float)
     - xo                  :int: Centroid (horizontal) position for cutting window
     - yo                  :int: Centroid (vertical) position for cutting window
     - x_size              :int: Horizontal cutting window size
     - y_size              :int: Vertical cutting window size
     - mask    :ndarray,ndarray: Tuple with index arrays (Output like numpy.where())

    Output:
     - img_cut  :ndarray: Resultant image array and (updated) header instance
    ---
    """
    
    xo = int(xo);
    yo = int(yo);
    x_size = int(x_size);
    y_size = int(y_size);

    imagem = image;
    
    # Initialize some variables..
    #
    x_diff = 0;   x_edge = 0;
    y_diff = 0;   y_edge = 0;
    x_fin = 0;   x_ini = 0;
    y_fin = 0;   y_ini = 0;

    y_img_size, x_img_size = imagem.shape;

    # Get the side sides (at least, 1!) and transform for the size_unit if necessary..
    #
    x_cut_size = max( 1, int(float(x_size)) );
    y_cut_size = max( 1, int(float(y_size)) );

    # And if no side size was given, define a default value correspondig to half of original image..
    #
    if not ( x_size ):
        x_cut_size = int(x_img_size/2);

    if not ( y_size ):
        y_cut_size = int(y_img_size/2);

    # Verify central coordinates values..
    #
    if (xo != 0):
        x_halo = int(float(xo));
    if (yo != 0):
        y_halo = int(float(yo));
    if (xo == 0):
        x_halo = int(x_img_size/2);
    if (yo == 0):
        y_halo = int(y_img_size/2);

    # Define the images (in/out) slices to be copied..
    #
    x_ini = x_halo - int(x_cut_size/2) #-1;
    x_fin = x_ini + x_cut_size;
    y_ini = y_halo - int(y_cut_size/2) #-1;
    y_fin = y_ini + y_cut_size;

    x_ini_old = max( 0, x_ini );   x_fin_old = min( x_img_size, x_fin );
    y_ini_old = max( 0, y_ini );   y_fin_old = min( y_img_size, y_fin );

    x_ini_new = abs( min( 0, x_ini ));   x_fin_new = x_cut_size - (x_fin - x_fin_old);
    y_ini_new = abs( min( 0, y_ini ));   y_fin_new = y_cut_size - (y_fin - y_fin_old);

    # Initialize new image, and take all index list..
    #
    imagemnova = np.zeros((y_cut_size,x_cut_size), dtype=imagem.dtype );
    ind_z = np.where(imagemnova == 0);

    # Copy requested image slice..
    #
    imagemnova[ y_ini_new:y_fin_new, x_ini_new:x_fin_new ] = imagem[ y_ini_old:y_fin_old, x_ini_old:x_fin_old ];

    # If 'mask', maintain just "central" object on it..
    #
    if ( mask ):
        msk = ( mask[0]-y_ini, mask[1]-x_ini )

        zip_m = zip( msk[0], msk[1] );
        zip_z = zip( ind_z[0], ind_z[1] );

        L = list(set(zip_z) - set(zip_m));

        try:
            ind_0, ind_1 = zip(*L);
            indx = ( np.array(ind_0), np.array(ind_1) );
            imagemnova[ indx ] = 0;
        except:
            pass;

    return imagemnova;
    
# ---
def poststamp(segimg,objID,objimg=None,increase=0,relative_increase=False,connected=False):
    """
    Identify objects on given images by their IDs and return object images

    By default, if 'objIDs' is not given, postamp will scan segmentation image 
    'seg_img' for the list of object ID numbers. If 'objIDs' is given, those IDs 
    will be used for object poststamps creation.

    'increase' and 'relative_increase' define whether the poststamps will have a 
    size different from object-only dimensions or just the object (pixels) itself, 
    and, also, if this increasement value is a multiplicative factor (True) or an 
    additive one (False).

    Since a list with object IDs is given, a list with arrays, with each IDentified
    object, is returned.

    Input:
     - segimg          ndarray : Segmented image (ndim=2,dtype=int)
     - objIDs            [int] : List with object IDs of interest in 'segimg'.
     - objimg          ndarray : Objects image (ndim=2,dtype=float)
     - increase          float : Value for poststamp resizing (> 0)
     - relative_increase  bool : Is 'increase' a additive value (default,False) or multiplicative one(True)?
        

    Output:
     - ndarray  : Image array and corresponding header
    ---
    """

    _id = objID;
    ind = np.where(segimg==objID);

    y_min = min( ind[0] );
    x_min = min( ind[1] );

    y_idx = ind[0] - y_min;
    x_idx = ind[1] - x_min;

    y_size = max( y_idx ) + 1;
    x_size = max( x_idx ) + 1;
    
   
    # Central pixel on original image:
    yo = y_size/2 + y_min;
    xo = x_size/2 + x_min;

    if ( increase != 0 ):		
        if (relative_increase == True):
            x_size = x_size*increase;
            y_size = y_size*increase;
        else:
            x_size = x_size + 2*increase;
            y_size = y_size + 2*increase;

    if objimg!=None:
        image_out = cutout(objimg,xo,yo,x_size,y_size,mask=ind);
    else:
        image_out = cutout(segimg,xo,yo,x_size,y_size,mask=ind);
    
    return image_out;

# ---
def mask(segimg,objID,objimg,negative=True,nullvalue=0):
    """
    Create a mask from values 'objID' in 'segimg' to select elements in 'objimg'.
    
    Given the two image arrays 'segimg' and 'objimg' with the same shape, the 
    elements in 'segimg' with values 'objID' are copied from 'objimg' to a new
    'output' image array. The 'output' array has the same shape as segimg/objimg.

    If 'negative=False', then the contrary is done: the elements valueing 'objID'
    in 'segimg' are excluded from the 'output' array containing the 'objimg' copy.
    
    Input:
     - segimg  ndarray : Segmented image array (ndim=2,dtype=int)
     - objID       int : Object ID to request output for
     - objimg  ndarray : Objects (observed) image array (ndim=2,dtype=float)
    
    Output:
     - output  ndarray : Same size (segimg) array with non-null pixels only for objID
    
    ---
    """

    outimg = np.zeros(segimg.shape,objimg.dtype) + nullvalue
    if negative:
        indxs = np.where(segimg==objID)
        outimg[indxs] = objimg[indxs]
    else:
        indxs = np.where(segimg!=objID)
        outimg[indxs] = objimg[indxs]
    
    return outimg


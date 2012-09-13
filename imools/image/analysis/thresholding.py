#
# Carlos Brandt - carloshenriquebrandt@gmail.com
# 4.* Imools - Out/2011
#
"""Image foreground|background threshold estimators."""

import numpy


# ---

def histmax(img):
    """
    Maximum histogram value for threshold estimation
    """
    np = numpy
    
    nbins=1000
    imhist,bins = np.histogram(img.flatten(),nbins,normed=True);

    return bins[np.argmax(imhist)]

# --

def riddler_calvard(img):
    """
    Riddler-Calvard method for image thrashold (fg|bg) estimation
    """
    import mahotas
    
    return mahotas.thresholding.rc(img)

rc = riddler_calvard

# --

def otsu(img):
    """
    Otsu method for image thrashold (fg|bg) estimation
    """
    import mahotas
    
    return mahotas.thresholding.ostu(img)

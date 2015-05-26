"""Image foreground|background threshold estimators."""

import numpy
import mahotas
import image

histmax = image.Stats.mode

def riddler_calvard(img):
    """
    Riddler-Calvard method for image thrashold (fg|bg) estimation
    """
    import mahotas
    
    return mahotas.thresholding.rc(img)

def otsu(img):
    """
    Otsu method for image thrashold (fg|bg) estimation
    """
    import mahotas
    
    return mahotas.thresholding.ostu(img)

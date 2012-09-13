# ---
def LeastSquare( x, y ):
    """Linear Weighted Regretion.
    Calculates the regretion of dataset around
    regions connected by linear relation.

    x (matrix) should be a column-aligned values array.
    y (matrix) should be a column vector.
    """

    import numpy as np;
    import sys;

    if( x.shape[0] != y.shape[0] ):
        print sys.stderr >> "Error: matrieces x,y have wrong dimensions.";
        return False;

    Xprod = x.T * x;
    Xinv = Xprod.I;
    Xprod = Xinv * x.T;

    theta = Xprod * y;

    return theta;


# ---
def LWR( x, y, xp, tau=1 ):
    """Linear Weighted Regretion.
    Calculates the regretion of dataset around
    regions connected by linear relation.

    x (array) should be a column-aligned values array.
    y (array) should be a column vector.
    """

    import numpy as np;
    from numpy.random import rand;
    import math;
    from math import exp;

    theta = rand(xp.shape[0]);

    if( x.shape[0] != y.shape[0] ):
        print sys.stderr >> "Error: matricies x,y have wrong dimensions.";
        return False;

    m = y.shape[0];
    w = np.zeros((m,m));
    for i in xrange(0,m):
        dist = x[i] - xp;
        w[i,i] = exp( -(np.inner(dist,dist)/tau**2 ) );
    XT = np.matrix(x).T;
    X = np.matrix(x);
    W = np.matrix(w);
    prod1 = XT * W * X;
    prod2 = XT * W * np.matrix(y).T;

    theta = prod1.I * prod2;
#    theta = prod2.I * prod1
    return np.inner(xp,np.array(theta.T))[0];


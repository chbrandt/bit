# ---
def GradienteEstocastico( x, y, alpha=0.01, eps=0.01 ):
    """Stocastic gradient descent.

    x (matrix) should be a column-aligned values array.
    y (matrix) should be a one-line array.
    alpha - [0:1] - is the learning rate.
    eps is the control error.
    """

    import sys;
    import numpy as np;
    from numpy.random import rand;

    m = y.shape[0];
    n = x.shape[1];
    theta = np.matrix(rand(n)).T;
    tp = 0;
    diff = 1;

    if( theta.size != x.shape[1] ):
        print sys.stderr >> "Error: matrix x is not using right dimensions.";
        return False;

    while( diff > eps ):
        for i in xrange(0,m):
            for j in xrange(0,n):
                J =  y[i] - x[i] * theta;
                theta[j] += alpha * ( J * (x[i].T)[j] );

        # Estima-se o desvio da parametrizacao relativo aos dados.
        tc = np.sum([ i**2 for i in theta ])**0.5;
        diff = abs( tc - tp );
        tp = tc;

    return theta;


def NB_multipar( X, Y ):

    import sys;
    import numpy;

    n_raws, n_cols = X.shape;
    
    x0 = X[numpy.where(Y==0)];
    x1 = X[numpy.where(Y==1)];
    n_raws0 = x0.shape[0];
    n_raws1 = n_raws - n_raws0;

    Prob_XY0 = numpy.zeros( (n_cols,3) );
    Prob_XY1 = numpy.zeros( (n_cols,3) );

    for i in xrange(0,n_cols):
        for j in range(3):
            Prob_XY0[i][j] = float(list(x0.T[i]).count(j+1))/n_raws0;
            Prob_XY1[i][j] = float(list(x1.T[i]).count(j+1))/n_raws1;

    prob_Y0 = float(n_raws0)/n_raws;
    prob_Y1 = float(n_raws1)/n_raws;

    return ( Prob_XY0, Prob_XY1, prob_Y0, prob_Y1 );

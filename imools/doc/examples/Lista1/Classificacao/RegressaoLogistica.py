
__date__ = "$ 26 October 2009 $"
__author__ = "Carlos Brandt"

def RegLog( x, y, alpha=0.001, eps=0.00001):

    import numpy as np;
    from numpy.random import rand;

    n = x.shape[1];
    m = x.shape[0];

    # Initialize theta vector
    theta = rand(n);
    thetal = np.zeros(n);
    # ... and auxiliar matrix...
    J = np.zeros(y.shape[0]);

    erro = 1;
    cnt = 0;

    while( erro > eps ):
        cnt += 1;
        thetal = theta.copy();

        for i in xrange(0,m):
            theta += alpha * np.dot(x[i], ( y[i] - 1/( 1 + np.exp(-1 * np.dot(x[i],theta)))) );

        diff = theta - thetal;
        erro = np.inner(diff,diff)
    
    print "\n Numero de iteracoes: %d\n Erro de parada: %.2E\n theta: %s\n" % (cnt,erro,theta);

    return theta;

#---
"""
    # Initialize theta vector
    theta = np.matrix(np.ones(n)).T;
    thetal = np.matrix(np.zeros(n)).T;
    # ... and auxiliar matrix...
    J = np.matrix(np.zeros(y.shape[0])).T;

    dif = theta - thetal;   # Take initial values difference
    difT = dif.T;           # transpose of 'dif'

    while( (difT * dif) > eps ):
        thetal = theta;
        for j in xrange(0,n):
            tmp = np.matrix([ exp(-1*float(x[i]*theta)) for i in xrange(len(x)) ]).T;
            J[:] = y[:] - 1/(1 + tmp[:]);
            theta[j] += alpha * np.sum(J * x.T[j]);

        dif = theta - thetal;
        difT = dif.T;

    return theta;
"""


__date__ = "$ 26 October 2009 $"
__author__ = "Carlos Brandt"

def NewtonRaphson( x, y, alpha=0.0000001, eps=0.01):

    import numpy as np;
    import math;
    from math import exp;

    n = x.shape[1];
    m = y.shape[0];

    # Initialize theta vector
    theta = np.ones((n,1));
    thetal = np.zeros((n,1));
    # ... and auxiliar matrix...
    J = np.zeros((m,1));

    h = np.zeros((n,n));
    gradl = np.zeros((n,1));

    dif = theta - thetal;   # Take initial values difference
    difT = dif.T;           # transpose of 'dif'

    while( np.dot(difT,dif) > eps ):
        thetal = theta;
        for i in range(0,n):
            taux = np.array([ np.dot(x[k],theta) for k in xrange(0,m) ]);
            gradl[i] = sum([ (y[k] - taux[k]) * x[k][i] for k in xrange(0,m) ]);   # Assumindo que h(x[k]) = x[k] * theta
            for j in range(0,n):
                h[i][j] = -1*sum([ taux[k] * (1 - taux[k]) * (x[k][i] * x[k][j]) for k in xrange(0,m) ]);

        Hinv = np.array(np.matrix(h).I);
        theta += -1*( np.dot(Hinv,gradl) );

        dif = theta - thetal;
        difT = dif.T;
#        erro = float(np.dot(difT,dif));
#        print theta
#        lixo = raw_input("lixo ")

    return theta;

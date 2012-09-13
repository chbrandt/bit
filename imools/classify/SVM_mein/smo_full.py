from __future__ import division;

__date__ = "Dec/2009";
__author__ = "Carlos Brandt";

def smo( X, Y, C, tol, max_passes ):
    """Sequential Minimal Optimization (SMO)

    Algorithm for training support vector machines.
    smo( X, : attributes training set
         Y, : classes [-1,1] training set
         C, : regularization parameter
         tol, : numerical tolerance
         max_passes : maximum number of iterations without changes
         )
         -> ( alpha, : list-like array (1D)
              b : threshold parameter)
    """

    import sys;
    import numpy as np;

    m,n = X.shape;

    alpha = np.zeros(m);
    b = 0;

    E = np.zeros(m);
    passes = 0;

    while( passes < max_passes ):

        num_changed_alphas = 0;

        for i in range(0,m):

            E[i] = b - Y[i] + sum([ alpha[k] * Y[k] * np.dot(X[k].T,X[i]) for k in range(0,m) ]);

            if( (Y[i]*E[i] < -tol and alpha[i] < C) or (Y[i]*E[i] > tol and alpha[i] > 0) ):

                j=i;
                while( j == i ):
                    j =int( m*np.random.rand());

                E[j] = -Y[j] + b + sum([ alpha[k] * Y[k] * np.dot(X[k].T,X[j]) for k in range(0,m) ]);
                alpha_i = alpha[i];
                alpha_j = alpha[j];

                if( Y[i] == Y[j] ):
                    L = max( 0, alpha_i + alpha_j - C );
                    H = min( C, alpha_i + alpha_j );
                else:
                    L = max( 0, alpha_j - alpha_i );
                    H = min( C, C + alpha_j - alpha_i );
                if( L == H ):
                    continue;

                ehta = 2*np.dot( X[i].T, X[j] ) - np.dot( X[i].T, X[i] ) - np.dot( X[j].T, X[j] );
                if( ehta >= 0 ):
                    continue;

                alpha[j] = alpha_j - Y[j] * (E[i] - E[j])/ehta;
                alpha[j] = H if (alpha[j] > H) else L if (alpha[j] < L) else alpha[j];
                if( abs(alpha[j]-alpha_j) < 1E-5 ):
                    continue;

                alpha[i] = alpha_i + Y[i]*Y[j] * ( alpha_j - alpha[j]);

                diff_ai = alpha[i] - alpha_i;
                diff_aj = alpha[j] - alpha_j;
                cross_dot = np.dot(X[i].T,X[j]);
                b_1 = b - E[i] - Y[i] * diff_ai * np.dot(X[i].T,X[i]) - Y[j] * cross_dot * diff_aj;
                b_2 = b - E[j] - Y[i] * diff_ai * cross_dot - Y[j] * np.dot(X[j].T,X[j]) * diff_aj;

                b = b_1 if (0 < alpha[i] < C) else b_2 if (0 < alpha[j] < C) else (b_1 + b_2)/2;

                num_changed_alphas += 1;

        if (num_changed_alphas == 0):
            passes += 1;
        else:
            passes = 0;

    return( alpha, b );

# ---

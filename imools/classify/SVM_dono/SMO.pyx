#!/usr/bin/env python

import sys;

import numpy as np;

# ==============
# Cython stuff..
#
cimport numpy as np;

ctypedef np.float64_t dtype_t

cdef extern from "cblas.h":
    double ddot "cblas_ddot"( int N,
                              double *X, int incX,
                              double *Y, int incY )

# ==============

# ---

def alpha_limits( alpha_i, alpha_j, C, y_i, y_j ):

    if (y_i != y_j):
        L = max( 0, alpha_j - alpha_i );
        H = min( C, C + alpha_j - alpha_i );
    else:
        L = max( 0, alpha_i + alpha_j - C );
        H = min( C, alpha_i + alpha_j );

    return (L, H);

# -

def E_deviation( np.ndarray[dtype_t, ndim=1] alpha,
                 dtype_t beta,
                 np.ndarray[dtype_t, ndim=2, mode="c"] X,
                 np.ndarray[dtype_t, ndim=1] Y,
                 np.ndarray[dtype_t, ndim=1, mode="fortran"] x_i,
                 dtype_t y_i ):

    cdef np.ndarray[dtype_t, ndim=1]   X_raw, x_col, fx_tmp

    x_col = x_i[:]
    for i in range( X.shape[0] ):
        X_raw = X[i,:]
        fx_tmp[i] = ddot( X_raw.shape[0], <dtype_t*>X_raw.data, X_raw.strides[0] // sizeof(dtype_t), <dtype_t*>x_col.data, x_col.strides[0] // sizeof(dtype_t) );

    fx = np.sum( alpha * Y * fx_tmp ) + beta;

#    fx = np.sum( alpha * Y * np.dot( X,x_i ) ) + beta;

    return (fx - y_i);

# -

def simple( C, tol, max_passes, X_tr, Y_tr ):
    """Simplified SMO

    Input:
     C : regularization parameter
     tol : numerical tolerance
     max_passes : max # of times to iterate without changes
     X_tr : array with examples parameters
     Y_tr : line array with exmaples classification (-1,1)

    Output:
     alpha : Lagrange multipliers for solution
     beta : solution threshold

    """


    num_exples, num_params = X_tr.shape;

    if (num_params < 2) :
        print "Error: Wrong training set.";
        return (False);

    alpha = np.zeros( num_exples );
    beta = 0;
    passes = 0;

    while (passes < max_passes):

        num_changed_alpha = 0;

        for i in range( 0, num_exples ):

            alpha_i = alpha[i];

            x_i = X_tr[i];
            y_i = Y_tr[i];
            E_i = E_deviation( alpha, beta, X_tr, Y_tr, x_i, y_i );

            if (( y_i*E_i < -1*tol and alpha_i < C ) or ( y_i*E_i > tol and alpha_i > 0 )):

                j = np.random.randint(num_exples);
                while ( j==i ): j = np.random.randint(num_exples);

                alpha_j = alpha[j];

                x_j = X_tr[j];
                y_j = Y_tr[j];
                E_j = E_deviation( alpha, beta, X_tr, Y_tr, x_j, y_j );

                alpha_i_old = alpha_i;
                alpha_j_old = alpha_j;

                L, H = alpha_limits( alpha_i, alpha_j, C, y_i, y_j );
                if ( L==H ):
                    continue;

#                eta = 2*np.inner( X_tr[i],X_tr[j] ) - np.inner( X_tr[i],X_tr[i] ) - np.inner( X_tr[j],X_tr[j] );
                eta = 2*np.dot( X_tr[i],X_tr[j] ) - np.dot( X_tr[i],X_tr[i] ) - np.dot( X_tr[j],X_tr[j] );
                if ( eta >= 0 ):
                    continue;

                alpha_j = alpha_j - y_j * ( ( E_i - E_j )/eta );

                if ( alpha_j > H ):
                    alpha[j] = H;
                elif ( alpha_j < L ):
                    alpha[j] = L;
                else:
                    alpha[j] = alpha_j;


                if ( abs( alpha_j-alpha_j_old ) < 1e-5 ):
                    continue;

                alpha[i] = alpha_i + ( y_i*y_j ) * ( alpha_j_old - alpha_j );

#                b1 = beta - E_i - y_i * (alpha_i - alpha_i_old) * np.inner(x_i,x_i) - y_j * (alpha_j - alpha_j_old) * np.inner(x_i,x_j);
#                b2 = beta - E_j - y_i * (alpha_i - alpha_i_old) * np.inner(x_i,x_j) - y_j * (alpha_j - alpha_j_old) * np.inner(x_j,x_j);
                b1 = beta - E_i - y_i * (alpha_i - alpha_i_old) * np.dot(x_i,x_i) - y_j * (alpha_j - alpha_j_old) * np.dot(x_i,x_j);
                b2 = beta - E_j - y_i * (alpha_i - alpha_i_old) * np.dot(x_i,x_j) - y_j * (alpha_j - alpha_j_old) * np.dot(x_j,x_j);

                if ( 0 < alpha_i < C ):
                    beta = b1;
                elif ( 0 < alpha_j < C ):
                    beta = b2;
                else:
                    (b1+b2)/2;

                num_changed_alpha += 1;

            # fi
        # rof

        if (num_changed_alpha == 0):
            passes += 1;
        else:
            passes = 0;

    # elihw

    return (alpha, beta);

# ---
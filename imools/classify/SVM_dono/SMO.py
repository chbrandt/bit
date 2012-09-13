#!/usr/bin/env python

import sys;

import numpy as np;
from numpy.random import randint as rand;


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

def kernel( raw, column ):
    return np.dot( raw, column );

# -

def E_deviation( alpha, beta, X, Y, x_i, y_i ):

    # A Kernel product should go into the next sentence:

    # option 1:
    fx = np.sum( alpha * Y * np.dot( X,x_i ) ) + beta;

    # option 2:
#    fx = sum( [ alpha[k] * Y[k] * np.dot( X[k],x_i ) for k in range( alpha.shape[0] ) ] ) + beta;

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

                j = rand(num_exples);
                while ( j==i ): j = rand(num_exples);

                alpha_j = alpha[j];

                x_j = X_tr[j];
                y_j = Y_tr[j];
                E_j = E_deviation( alpha, beta, X_tr, Y_tr, x_j, y_j );

                alpha_i_old = alpha_i;
                alpha_j_old = alpha_j;

                L, H = alpha_limits( alpha_i, alpha_j, C, y_i, y_j );
                if ( L==H ):
                    continue;

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


##                if ( abs( alpha_j-alpha_j_old ) < 1e-5 ):
##                    continue;

                alpha[i] = alpha_i + ( y_i*y_j ) * ( alpha_j_old - alpha_j );

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

# -

def full( C, tol, max_passes, X_tr, Y_tr ):
    """Full SMO algorithm

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
    E = np.zeros( num_exples );
    passes = 0;

    while (passes < max_passes):

        num_changed_alpha = 0;

        for i in range( 0, num_exples ):

            alpha_i = alpha[i];
            x_i = X_tr[i];
            y_i = Y_tr[i];

            E[i] = E_deviation( alpha, beta, X_tr, Y_tr, x_i, y_i );

            if (( y_i*E[i] < -1*tol and alpha_i < C ) or ( y_i*E[i] > tol and alpha_i > 0 )):

                if ( len( np.where( alpha % C )) != 0 ):
                    # Faz uma escolha por i
                    pass;

                j = rand(num_exples);
                while ( j==i ): j = rand(num_exples);

                alpha_j = alpha[j];
                x_j = X_tr[j];
                y_j = Y_tr[j];

                E[j] = E_deviation( alpha, beta, X_tr, Y_tr, x_j, y_j );

                # Compute L, H..
                #
                L, H = alpha_limits( alpha_i, alpha_j, C, y_i, y_j );
                if ( L==H ):
                    continue;

                # 
                eta = 2*kernel( X_tr[i],X_tr[j] ) - kernel( X_tr[i],X_tr[i] ) - kernel( X_tr[j],X_tr[j] );
                if ( eta >= 0 ):
                    continue;

                alpha_i_old = alpha_i;
                alpha_j_old = alpha_j;

                alpha_j = alpha_j - y_j * ( ( E_i - E_j )/eta );

                if ( alpha_j > H ):
                    alpha[j] = H;
                elif ( alpha_j < L ):
                    alpha[j] = L;
                else:
                    alpha[j] = alpha_j;


                if ( abs( alpha_j-alpha_j_old ) < tol*(alpha_j + alpha_j_old + tol) ):
                    continue;


                alpha[i] = alpha_i + ( y_i*y_j ) * ( alpha_j_old - alpha_j );

                b1 = beta - E[i] - y_i * (alpha_i - alpha_i_old) * kernel(x_i,x_i) - y_j * (alpha_j - alpha_j_old) * kernel(x_i,x_j);
                b2 = beta - E[j] - y_i * (alpha_i - alpha_i_old) * kernel(x_i,x_j) - y_j * (alpha_j - alpha_j_old) * kernel(x_j,x_j);

                if ( 0 < alpha_i < C ):
                    beta = b1;
                elif ( 0 < alpha_j < C ):
                    beta = b2;
                else:
                    (b1+b2)/2;

                E[i] = E_deviation( alpha, beta, X_tr, Y_tr, x_i, y_i );
                E[j] = E_deviation( alpha, beta, X_tr, Y_tr, x_i, y_i );

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

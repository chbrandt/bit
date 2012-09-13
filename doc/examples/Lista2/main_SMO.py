#!/usr/bin/env python
from __future__ import division;

def be( W, X, Y ):
    import numpy as np;

    b0 = -999;
    b1 = 999;
    for i in range(0,len(Y)):
        if Y[i] == -1 :
            b0 = max(b0,np.inner(W,X[i]));
        else:
            b1 = min(b1,np.inner(W,X[i]));
    b = -b0/2 + b1;
    return (b);

# ---

def split_dataset( A, i, n ):
    import numpy as np;

    Nraws = A.shape[0];
    Atr1 = A[0:Nraws*i/n];
    Ats = A[Nraws*i/n:Nraws*(i+1)/n];
    Atr2 = A[Nraws*(i+1)/n:Nraws];
    Atr = np.concatenate((Atr1,Atr2),axis=0);

    return (Atr,Ats);

# ---

def run_smo( X, Y, C, tolerance, max_passes ):
#    from SMO import smo;
    from smo_v1 import smo;
    import numpy as np;

    num_TestSets = 3;
    num_TestSets = max(2,num_TestSets)
    n = num_TestSets;

    N_raws = X.shape[0];
    N_miss_classified = N_miss_old = N_raws;

    for i in range(n):
        Xtr,Xts = split_dataset(X,i,n);
        Ytr,Yts = split_dataset(Y,i,n);

        print "Running SMO... %d of %d." % (i+1,n);
        alpha, b = smo( Xtr, Ytr, C, tolerance, max_passes );

        W = np.zeros( Xtr.shape[1] );
        for j in range( 0, Xtr.shape[0] ):
            W += alpha[j] * Ytr[j] * Xtr[j];

#        blinha = be(W,Xtr,Ytr);
        print "alpha: ",alpha;
        print "b: ",b;
#        print blinha;
        
        print "W",W

        Res = np.dot( W,Xts.T ) + b;
        print i+1,") ", Res/Yts;

        N_miss_classified = len( np.where(Res/Yts < 0)[0] );
#        if (N_miss_classified <= N_miss_old):
#            W_res = W;
#            b_res = b

        W_res = W
        b_res = b
    return(W_res,b_res);

# ---

if __name__ == "__main__" :
    import sys;
    import numpy as np;

    C = 1;
    tolerance = 0.001;
    max_passes = 10;

    if( len(sys.argv) < 3 ):
        print >> sys.stderr, "Usage:\
    %s <arguments training set> <class training set> [arguments test set]" % (sys.argv[0]);
        exit(99);

    X = np.loadtxt(sys.argv[1]);
    Y = np.loadtxt(sys.argv[2]);

    W,b = run_smo( X, Y, C, tolerance, max_passes );

    print "Final results for W and b:";
    print W;
    print b;

    Res = np.dot( W, X.T ) + b;
    print (Res/np.abs(Res))/Y;
    print "Missclassified:",size( np.where( (Res/np.abs(Res))/Y == -1 )[0] )/size( (Res/np.abs(Res))/Y )

    if ( len(sys.argv)==4 ):
        X_tst = np.loadtxt(sys.argv[3]);
        Res = np.dot( W, X_tst.T ) + b;
        print "classification of %s:" % (sys.argv[3]);
        print Res/np.abs(Res);
        np.savetxt('diagnostico_ts_Y_ghama.dat',Res);
        np.savetxt('diagnostico_ts_Y_class.dat',Res/np.abs(Res));

    exit(0);

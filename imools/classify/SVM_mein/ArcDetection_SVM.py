#!/usr/bin/env python
from __future__ import division;
# ---

def fits_array_size( input_file ):
    import pyfits;

    fp = open( input_file );
    biggest_image_size = 0;
    for _each in fp.readlines():
        _each.rstrip("\n");
        hdr = pyfits.getheader(_each);
        naxis1 = hdr['NAXIS1'];
        naxis2 = hdr['NAXIS2'];
        biggest_image_size = max( biggest_image_size, naxis1*naxis2 );

    return biggest_image_size;
# --

def fits2vec( filename, size=0 ):
    import pyfits;

    img = pyfits.getdata( filename );
    m,n = img.shape;
    dim = m*n
    img = img.reshape((1,dim));

    if (size):
        diff = size - dim;
        halfadd = diff/2;
        rest = diff%2;
        _img = np.concatenate((np.zeros((1,halfadd+rest)),img,np.zeros((1,halfadd))),axis=1);
        img = _img.copy();

    return img;
# --

# --- MAIN --- #
if __name__ == "__main__" :

    """Gravitational Arcs recognition procedure.

    This program opens a set of FITS images of
    simulated gravitational arcs ( as poststamps)
    and tries to train a SVM for future recognition.

    Input: text file with FITS filenames.

    """

    import sys;
    import numpy as np;
    from numpy.random import poisson;

    from main_SMO import run_smo;

    C = 1;
    tolerance = 0.001;
    max_passes = 10;

    if len(sys.argv) < 3 :
        print >> sys.stderr, "Usage:\
    %s <arguments training set> <class training set> [arguments test set]" % (sys.argv[0]);
        exit(99);

    input_file_X = sys.argv[1];
    input_file_Y = sys.argv[2];

    biggest_image_size = fits_array_size(input_file_X);

    fp = open( input_file_X,'r' );

    X = fits2vec( fp.readline().rstrip("\n"), size=biggest_image_size )
    for iFits in fp.readlines():
        img1D = fits2vec( iFits, size=biggest_image_size );
        X = np.concatenate((X,img1D),axis=0);

    Y = np.loadtxt(input_file_Y);

    # Separar o set de treinamento para posterior teste..
    m,n = X.shape;
    Xtr = X[:m*7/9];
    Ytr = Y[:m*7/9];
    Xts = X[m*7/9:];
    Yts = Y[m*7/9:];

    Xts = poisson(Xts);
    
    W,b = run_smo( Xtr, Ytr, C, tolerance, max_passes );

    print "Final results for W and b:";
    print W;
    print b;

    Res = np.dot( W, Xts.T ) + b;
    print (Res/np.abs(Res))/Yts;
    print "Missclassified:",np.where( (Res/np.abs(Res))/Yts == -1 )[0].size/((Res/np.abs(Res))/Yts).size;

    if ( len(sys.argv)==4 ):
        X_tst = np.loadtxt(sys.argv[3]);
        Res = np.dot( W, X_tst.T ) + b;
        print "classification of %s:" % (sys.argv[3]);
        print Res/np.abs(Res);
        np.savetxt('diagnostico_ts_Y_ghama.dat',Res);
        np.savetxt('diagnostico_ts_Y_class.dat',Res/np.abs(Res));

    exit(0);

#!/usr/bin/env python
import sys;

import pyfits;

# ---

def fits2vec( filename, size=0 ):

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

# -

def fitsFiles_2_array( file, image_size ):

    fp = open( file, 'r' );

    X = fits2vec( fp.readline().rstrip("\n"), size=image_size )
    for iFits in fp.readlines():
        img1D = fits2vec( iFits, size=image_size );
        X = np.concatenate((X,img1D),axis=0);

    return X;

# -

def biggest_fits( input_file ):
    """Verify the biggest array given a list of them (images)

    Input:
     Ascii file with list of FITS filenames

    Output:
     Size of the biggest one

    """
    fp = open( input_file );

    biggest_size = 0;

    for _each in fp.readlines():
        _each.rstrip("\n");
        hdr = pyfits.getheader(_each);
        naxis1 = hdr['NAXIS1'];
        naxis2 = hdr['NAXIS2'];
        biggest_size = max( biggest_size, naxis1*naxis2 );

    return biggest_size;

# ---

# =========================
if __name__ == "__main__" :

    import numpy as np;
    from numpy.random import poisson;

    import SMO;

    import optparse;
    import pstats;
    import cProfile;


    # Option parser:
    parser = optparse.OptionParser();

    parser.add_option("-X","--examples_list",
                      dest='trainXset',default=None,
                      help="List of FITS image filenames (training set)");
    parser.add_option("-Y","--examples_class",
                      dest='trainYset',default=None,
                      help="Ascii file with examples classification (-1,1)");
    parser.add_option("-Z","--test_set",
                      dest='testset',default=None,
                      help="List of FITS image names for classification test");
    parser.add_option("-n","--max_passes",
                      dest='Npass', default='10',
                      help="Max number of loops over same optimal state");
    parser.add_option("-t","--tol",
                      dest='tol',default='0.1',
                      help="Tolerance parameter for (optimal) solution");
    parser.add_option("-C","--reg_param",
                      dest='C',default='10',
                      help="Regularization parameter (C), 0 <= alpha[] <= C");

    (opts,args) = parser.parse_args();

    if ( opts == {} ):
        parser.print_help();
        sys.exit(0);

    # Files (list) with the sets of images..
    #
    X_fitsList = opts.trainXset;
    Y_clasList = opts.trainYset;
    test_set = opts.testset;

    # SMO parameters..
    #
    max_passes = int(opts.Npass);
    tolerance = float(opts.tol);
    C = float(opts.C);

    if ( X_fitsList==None or Y_clasList==None ):
        print >> sys.stderr,"Error: Enter both files for training set";
        sys.exit(2);

    # Get the biggest image size..
    #
    biggest_size = biggest_fits( X_fitsList );
    print "Major image: %s pixels" % (biggest_size);

    # Open training set of images and their class info..
    #
    print "Opening training set...";
    X = fitsFiles_2_array ( X_fitsList, biggest_size );
    Y = np.loadtxt( Y_clasList );

    # Split training set for validation..
    #
    m,n = X.shape;
    Xtr = X[:m*7/9];
    Ytr = Y[:m*7/9];
    Xts = X[m*7/9:];
    Yts = Y[m*7/9:];

    num_exples, num_params = Xtr.shape;

#    Xts = poisson(Xts);

    # Run SMO (simplified) algorithm for SVM training..
    #
    print "Running SMO...";

    cProfile.run( 'alpha, beta = SMO.simple( C, tolerance, max_passes, Xtr, Ytr )', 'stats.prof');

    print "Done.";

    p = pstats.Stats('stats.prof');
#    p.strip_dirs().sort_stats('cumulative').print_stats();
    p.sort_stats('cumulative').print_stats();

#    print "Alpha: ", alpha;
    print "Beta: ", beta;

    # Built-up the W operator..
    #
    W = np.zeros( num_params );
    for j in range( 0, num_exples ):
        W += alpha[j] * Ytr[j] * Xtr[j];

    print "Final results for W and beta:";
#    print W;
#    print beta;

    Res = np.dot( W, Xts.T ) + beta;
    print (Res/np.abs(Res))/Yts;
    print "Missclassified: ",float(np.where( (Res/np.abs(Res))/Yts == -1 )[0].size)/((Res/np.abs(Res))/Yts).size;

#    if ( len(sys.argv)==4 ):
#        X_tst = np.loadtxt(sys.argv[3]);
#        Res = np.dot( W, X_tst.T ) + b;
#        print "classification of %s:" % (sys.argv[3]);
#        print Res/np.abs(Res);
#        np.savetxt('diagnostico_ts_Y_ghama.dat',Res);
#        np.savetxt('diagnostico_ts_Y_class.dat',Res/np.abs(Res));


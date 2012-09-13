#!/usr/bin/env python

if __name__ == "__main__":

    import sys;
    import numpy;
    import pylab;
    from NaiveBayes import NB_multipar;

    if ( len(sys.argv) != 4 ):
        print >> sys.stderr, "Usage:\n %s <X-training-file> <Y-training-file> <X-test-file>" % (sys.argv[0]);
        exit (False);

    # Read filenames
    fxtr = sys.argv[1];   # x training (counts)
    fytr = sys.argv[2];   # y training (class)
    ftst = sys.argv[3];   # test file  (counts to classify)

    X_training = pylab.load( fxtr );
    Y_training = pylab.load( fytr );

    if ( X_training.shape[0] != Y_training.shape[0] ):
        print >> stderr, "Sizes of training set do not match.\n Verify the number of lines."
        exit (False);

    Prob_XY0, Prob_XY1, prob_Y0, prob_Y1 = NB_multipar( X_training, Y_training );

    print "\n Prob. y=0: %.5f\n Prob. y=1: %.5f" % (prob_Y0,prob_Y1);
    print " P(y=0) + P(y=1): %.5f\n" % (prob_Y0 + prob_Y1);
    pylab.save('Probability_matrix_Y0.dat', Prob_XY0, fmt='%.5e');
    pylab.save('Probability_matrix_Y1.dat', Prob_XY1, fmt='%.5e');

    X_test = pylab.load( ftst );

    n_raws = X_test.shape[0];
    n_cols = X_test.shape[1];

    Prob_class_Y0 = numpy.zeros(n_cols);
    Prob_class_Y1 = numpy.zeros(n_cols);
    Class_result = numpy.zeros((n_raws,1));

    for i in xrange( n_raws ):

        for j in xrange( n_cols ):
            Prob_class_Y0[j] = Prob_XY0[j][ X_test[i][j]-1 ];
            Prob_class_Y1[j] = Prob_XY1[j][ X_test[i][j]-1 ];

        if ( Prob_class_Y0.prod()*prob_Y0 > Prob_class_Y1.prod()*prob_Y1 ):
            Class_result[i] = 0;
        else:
            Class_result[i] = 1;

    Mout = numpy.concatenate((X_test,Class_result),axis=1);

#    print "Classification results: ", Class_result.T;
    pylab.save('Classification_results.dat', Mout, fmt='%.1e');
    print "Look into file 'Classification_results.dat'.\n";

    exit (True);

#!/usr/bin/env python

import sys;
import numpy as np;
import scipy;
import pylab;
from RegressaoLogistica import RegLog;

print sys.argv;

if( len(sys.argv) != 3 ):
    print "\nUsage:\n       # python %s data-points.txt class.txt\n" % (sys.argv[0])
    exit();

fx = sys.argv[1];
fy = sys.argv[2];
x = pylab.load(fx);   # array
y = pylab.load(fy);   # array
xmax = int(max(x.T[0]));
xmin = int(min(x.T[0]));

xm = np.concatenate( ( np.ones((x.shape[0],1)), x ), axis=1 );

theta = RegLog(xm,y,alpha=0.001,eps=0.000001);

print "fit: y = %fx + %f" % ( -theta[1]/theta[2], -theta[0]/theta[2] );

# Plot neles!
x0 = x[np.where(y==0)[0]];
x1 = x[np.where(y==1)[0]];

xd = np.arange(xmin-1,xmax+1,0.1);
theta = theta.tolist();
#theta.reverse();
#yfit = scipy.polyval(theta,xd);
yfit = -( theta[1]*xd + theta[0] ) / theta[2];

pylab.plot(x0.T[0],x0.T[1],'mo');
pylab.plot(x1.T[0],x1.T[1],'b^');
pylab.plot(xd,yfit,'r-');

pylab.title('Divisao de classes');
pylab.xlabel('X');
pylab.ylabel('Y');

#pylab.show();
fout = fx + '_OUT_.png'
pylab.savefig( fout, format='png' );


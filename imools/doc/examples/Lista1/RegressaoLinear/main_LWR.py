#!/usr/bin/env python

import LWR;
from LWR import LWR;
import numpy as np;
import scipy;
import pylab;

#---
def erroquad(yFit,yReal):
    m = len(yReal);
    return ( sum([ (yFit[i]-yReal[i])**2 for i in xrange(0,m) ]) / m );
#---

Fx1 = open('reg_2_tr_X.dat','r');
Fy1 = open('reg_2_tr_Y.dat','r');
Fx2 = open('reg_2_ts_X.dat','r');
xl = [ float(line.rstrip('\r\n')) for line in Fx1.readlines() ];
yl = [ float(line.rstrip('\r\n')) for line in Fy1.readlines() ];
xtl = [ float(line.rstrip('\r\n')) for line in Fx2.readlines() ];

x = np.array([ np.ones(len(xl)), xl ]).T;
y = np.array(yl).T;
xt = np.array([ np.ones(len(xtl)), xtl ]).T;

del xtl;

yf = [];
xf = [];
for xp in x:
    yf.append( LWR(x,y,xp) );
    xf.append( xp[1] );

#print "fit Lote: %f + %fx" % (thetaL[0],thetaL[1]);

#erro = erroquad(yfitL,yd)
pnts = zip(xf,yf);
pnts.sort();
xf,yf = zip(*pnts);

pylab.plot(xl,yl,'o');
pylab.plot(xf,yf,'r-');
pylab.xlabel('X')
pylab.ylabel('Y')
pylab.title('Ajuste linear ponderado dos pontos')

#print erro;
# Plota o conjunto de teste:
#Fxt = open('reg_1_ts_X.dat','r');
#xt = [ float(line.rstrip('\r\n')) for line in Fxt.readlines() ];
#yfitt = scipy.polyval(thetaS,xt);
#pylab.plot(xt,yfitt,'ro');

#pylab.show()
pylab.savefig('reg_2_plot_reglin.png',format='png');


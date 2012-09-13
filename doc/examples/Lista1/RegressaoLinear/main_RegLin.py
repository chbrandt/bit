#!/usr/bin/env python

import RegressaoLinear as RL;
import numpy as np;
import scipy;
import pylab;

#---
def erroquad(yFit,yReal):
    m = len(yReal);
    return ( sum([ (yFit[i]-yReal[i])**2 for i in xrange(0,m) ]) / m );
#---

Fx1 = open('reg_1_tr_X.dat','r');
Fy1 = open('reg_1_tr_Y.dat','r');
xl = [ float(line.rstrip('\r\n')) for line in Fx1.readlines() ];
yl = [ float(line.rstrip('\r\n')) for line in Fy1.readlines() ];

x = np.matrix([ np.ones(len(xl)), xl ]).transpose();
y = np.matrix(yl).T;

del xl;
del yl;

thetaL = RL.GradienteLote(x,y);
#thetaE = RL.GradienteEstocastico(x,y);
#thetaS = RL.LeastSquare(x,y);

print "fit Lote: %f + %fx" % (thetaL[0],thetaL[1]);
#print "fit Estoc: %fx + %f" % (thetaE[0],thetaE[1]);
#print "fit LMS: %fx + %f" % (thetaS[0],thetaS[1]);

# Plot neles!
yd = (y.T).tolist()[0];
xd = (x.T[-1]).tolist()[0];

thetaL = [ float(thetaL[i]) for i in range(len(thetaL)-1,-1,-1) ];
#thetaE = [ float(thetaE[i]) for i in range(len(thetaE)-1,-1,-1) ];
#thetaS = [ float(thetaS[i]) for i in range(len(thetaS)-1,-1,-1) ];
yfitL = scipy.polyval(thetaL,xd);
#yfitE = scipy.polyval(thetaE,xd);
#yfitS = scipy.polyval(thetaS,xd);

erro = erroquad(yfitL,yd)

pylab.plot(xd,yd,'o');
pylab.plot(xd,yfitL,'r-');
pylab.xlabel('X')
pylab.ylabel('Y')
pylab.title('Ajuste linear dos pontos')
#pylab.plot(xd,yfitE,'b-.');
#pylab.plot(xd,yfitS,'k-');

print erro;
# Plota o conjunto de teste:
#Fxt = open('reg_1_ts_X.dat','r');
#xt = [ float(line.rstrip('\r\n')) for line in Fxt.readlines() ];
#yfitt = scipy.polyval(thetaS,xt);
#pylab.plot(xt,yfitt,'ro');

pylab.show()
#pylab.savefig('reg_1_plot_reglin.png',format='png');


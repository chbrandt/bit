import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import time
from evolution import *
from plot_u import *

def gauss_kern():
    """ Returns a normalized 2D gauss kernel array for convolutions """
    h1 = 15
    h2 = 15
    x, y = np.mgrid[0:h2, 0:h1]
    x = x-h2/2
    y = y-h1/2
    sigma = 1.5
    g = np.exp( -( x**2 + y**2 ) / (2*sigma**2) );
    return g / g.sum()

#Img = plt.imread("twoObj.bmp")
Img = plt.imread("ds9.bmp")
Img = Img[::-1] 
g = gauss_kern()
Img_smooth = signal.convolve(Img,g,mode='same')
Iy,Ix=np.gradient(Img_smooth)

f=Ix**2+Iy**2
g=1. / (1.+f)  # edge indicator function.
epsilon=1.5 # the papramater in the definition of smoothed Dirac function
timestep=5  # time step
mu=0.2/timestep  # coefficient of the internal (penalizing) 
                  # energy term P(\phi)
                  # Note: the product timestep*mu must be less 
                  # than 0.25 for stability!

lam=5 # coefficient of the weighted length term Lg(\phi)
alf=3 # coefficient of the weighted area term Ag(\phi);
      # Note: Choose a positive(negative) alf if the 
      # initial contour is outside(inside) the object.

nrow, ncol=Img.shape

c0=4

initialLSF=c0*np.ones((nrow,ncol))

w=8

initialLSF[w+1:-w-1, w+1:-w-1]=-c0

u=initialLSF

#plot_u(u)

plt.ion()

for n in range(300):    
    u=evolution(u, g ,lam, mu, alf, epsilon, timestep, 1)
    if np.mod(n,20)==0:        
        #plot_u(u)
        plt.imshow(Img, cmap='gray')
        plt.hold(True)
        CS = plt.contour(u,0, colors='r') 
        plt.draw()
        time.sleep(1)
        plt.hold(False)
        plt.savefig('out_%s.png' % n)

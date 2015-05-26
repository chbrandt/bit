import numpy as np
from scipy import ndimage
import scipy.ndimage.filters

def NeumannBoundCond(f):
    # Make a function satisfy Neumann boundary condition
    nrow,ncol = f.shape
    g = f

    g[0,0] = g[2,2]
    g[0,ncol-1] = g[2,ncol-3]
    g[nrow-1,0] = g[nrow-3,2]
    g[nrow-1,ncol-1] = g[nrow-3,ncol-3]

    g[0,1:-1] = g[2,1:-1]
    g[nrow-1,1:-1] = g[nrow-3,1:-1]

    g[1:-1,0] = g[1:-1, 2]
    g[1:-1,ncol-1] = g[1:-1,ncol-3]
    
    return g

def Dirac(x, sigma):
    f=(1./2./sigma)*(1.+np.cos(np.pi*x/sigma))
    b = (x<=sigma) & (x>=-sigma)
    f = f*b;
    return f

def curvature_central(nx,ny):
    [junk,nxx]=np.gradient(nx)
    [nyy,junk]=np.gradient(ny)
    K=nxx+nyy
    return K

def evolution(u0, g, lam, mu, alf, epsilon, delt, numIter):
    u=u0
    vy,vx=np.gradient(g)
    for k in range(numIter):
        u=NeumannBoundCond(u)
        [uy,ux]=np.gradient(u)
        normDu=np.sqrt(ux**2 + uy**2 + 1e-10)
        Nx=ux/normDu
        Ny=uy/normDu
        diracU=Dirac(u,epsilon)
        K=curvature_central(Nx,Ny)
        weightedLengthTerm=lam*diracU*(vx*Nx + vy*Ny + g*K)        
        penalizingTerm=mu*(scipy.ndimage.filters.laplace(u)-K)
        weightedAreaTerm=alf*diracU*g
        u=u+delt*(weightedLengthTerm + weightedAreaTerm + penalizingTerm)  # update the level set function
        
    return u

import numpy as np

# --

def grid(X=(-5,5,0.1),Y=(-5,5,0.1)):
    """
    Returns the pair X,Y of grid coordinates
    
    Input:
     - shape  (int,int) : Sizes of X,Y sides
     - res        float : point resolution
    
    Output:
     - X,Y  ndarrays : x,y meshgrids
    
    ---
    """
    
    X = list(X)
    Y = list(Y)
    X[1] += X[2]
    Y[1] += Y[2]
    X = np.arange(*X)
    Y = np.arange(*Y)
    X,Y = np.meshgrid(X,Y)

    return X,Y
    
# --

def sersic(R=0,n=0.5,b=1,X=(-5,5,0.1),Y=(-5,5,0.1)):
    """
    Returns an intensity array of a Sersic profile
    
    Input:
    
    Output:
    
    ---
    """
    
    X,Y = grid(X,Y)

    if not R:
        R = np.sqrt(X**2 + Y**2)
        R = 2*R/R.max()
    else:
        R = np.sqrt(X**2 + Y**2)/R

    Z = np.exp( -b * (np.power(R,1./n) - 1))

    return Z


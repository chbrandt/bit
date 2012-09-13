# ---
def GradienteLote( x, y, alpha=0.01, eps = 0.01 ):
    """Batch gradient descent.

    x (matrix) should be a column-aligned values array.
    y (matrix) should be a one-line array.
    alpha - [0:1] - is the learning rate.
    eps is the control error.
    """

    import sys;
    import numpy as np;
    from numpy.random import rand;

    # Le-se algumas dimensoes...
    m = y.shape[0];
    n = x.shape[1];

    # e inicializa-se algumas variaveis de controle e ajuste.
    theta = np.matrix(rand(n)).T;

    tp = 0;
    diff = 1;
    J = np.matrix(np.zeros(m)).T;

    if( theta.size != x.shape[1] ):
        print sys.stderr >> "Error: array x is not using right dimensions.";
        exit();

    while( diff > eps ):
        # Nesse trecho tem loop implicito; a sintaxe '[:]' ajuda a identificar...
        inn = (x * theta);
        J[:] = y[:] - inn[:];
        Jm = np.sum(J);
        # Apos a computacao da distancia entre saida real e estimada, ajusta-se os parametros..
        for j in xrange(0,n):
            theta[j] += alpha * (( J.T * x ).T)[j];
        # Estima-se o desvio da parametrizacao relativo aos dados.
        tc = np.sum([ i**2 for i in theta ])**0.5;
        diff = abs( tc - tp );
        tp = tc;

    return theta;


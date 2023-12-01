import numpy as np

def gauss_iter_solve(A,b,x0,tol,alg):
    """
    Description of function.
        Parameters
        A: Coefficient Matrix (nxn) (array-like) 
        b: Right Hand Side Vector (mxn) (array-like)
        x0: optional initial guess, defaults to all 0s (array-like)
        tol: error tolerance, defaults to 1e-8 (mxn) (float)
        alg: optional choice of algorithm, defaults to seidel (string)
        ----------
        Returns
        
        -------
    """
    A=np.array(A)
    b=np.array(b)

    if alg.strip().lower()!=('seidel' or 'jacobi'):
        raise ValueError("enter valid algorithm (seidel or jacobi)")
    n=len(A)
    if A.shape[1]!=n:
        raise ValueError("A is not square, has {n} rows and {A.shape[1]} columns")
    dimb=len(b.shape)
    if dimb!=1:
        raise ValueError("b has {dimb} dimensions, should be 1")
    
    identity=[[] for i in range(n)]
    for i in range (0,n,1):
        for j in range (0,n,1):
            if i==j:
                identity[i].append(1)
            else:
                identity[i].append(0)

    A_diag=[[] for i in range (n)]
    for i in range (0,n,1):
        for j in range (0,n,1):
            if i==j:
                #inverse of A is calculated here to save time and memory
                A_diag[i].append(1/A[i][j])
            else:
                A_diag[i].append(0)

    #calculation of the normalized matrices
    A_star=np.matmul(A_diag,A)

    b_star=np.matmul(A_diag,b)

    A_s=np.subtract(A_star,identity)

    k=0

    #if x_init==True:
    #    x0=x0
        #maybe just make this first and then update if there is a x0

    diff_x=1

    max_iter=5

    N=0

    if alg=="seidel":
        while (N<max_iter):
            x_old=x0.copy()
            for i in range(0,n,1):
                #print()
                A_k1=sum(A_star[i][j]*x0[j] for j in range (0,n) )
                A_k=sum(A_star[i][j]*x_old[j] for j in range (0,n) )
                #print("A_k1", A_k1)
                #print("A_k", A_k)
                x0[i]=(b_star[i]-A_k1-A_k)
                #print("x0", x0)
            N+=1
            if all(abs(x0[i]-x_old[i])<tol for i in range(n)):
                return(x0,N)
            if (N>max_iter):
                raise RuntimeError
        #return(x0,N)
    elif alg=="jacobi":
        pass
        #inverse of main diagonal A (B-A_s * X)

def spline_function(xd,yd,order):
     """
    Description of function.
        Parameters
        xd:
        yd:
        order:
        ----------
        Returns
        
        -------
    """
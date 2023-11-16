def gauss_iter_solve(A,b,x0,tol,alg):
    """
    Description of function.
        Parameters
        A: Coefficient Matrix (nxn) (array-like) 
        b: Right Hand Side Vector (mxn) (array-like)
        x0: optional initial guess, defaults to 0 (array-like)
        tol: error tolerance, defaults to 1e-8 (mxn) (float)
        alg: optional choice of algorithm, defaults to seidel (string)
        ----------
        Returns
        
        -------
    """
    n=len(A)
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
                A_diag[i].append(1/A[i][j])
            else:
                A_diag[i].append(0)
    
    #A_s=A-A_diag
    k=0
    if x_init==False:
        for n in range (0,n,1):
            for m in range (0,m,1):
                x0.append[0]
    if alg=="seidel":
        while (#sumofx<tol):
            for i in range(0,n,1):
                for j in range (0,n,1):
                    x_k+=x*A[i][j]
                    
                    #inverse of main diagonal A (B-A_s * X)
    elif alg=="jacobi":
        pass
    else:
        raise ValueError

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
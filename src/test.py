from linalg_interp import gauss_iter_solve
import numpy as np

A=np.array([
    [4,-1,0,0],
    [-1,4,-1,0],
    [0,-1,4,-1],
    [0,0,-1,3]
])

b=np.array(
    [15,10,10,10]
)

x0=[0,0,0,0]

tol=1e-8

alg="seidel"

x0,N=gauss_iter_solve(A,b,x0,tol,alg)

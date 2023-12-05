import numpy as np
import matplotlib.pyplot as plt

from lab02.linalg_interp import (
    gauss_iter_solve,
    spline_function,
)

def test_gauss_iter_solve():
    A=np.array([
        [4,-1,0,0],
        [-1,4,-1,0],
        [0,-1,4,-1],
        [0,0,-1,3]
    ])

    b=np.array(
        [15,10,10,10]
    )


    x0=gauss_iter_solve(A,b)

    x_exp=np.linalg.solve (A,b)

    print("gauss seidel")
    print(f"expected:\n{x_exp}")
    print(f"actual:\n{x0}")

    xj=gauss_iter_solve(A,b,alg="jacobi")

    print("jacobi")
    print(f"expected:\n{x_exp}")
    print(f"actual:\n{xj}")


def test_spline():
    xd=np.linspace(-5.0,5.0,10)
    yd=1.0+0.5*xd

    s1=spline_function(xd,yd,order=1)
    xp=np.linspace(-6.0,6.0,100)
    yp_exp=1.0+0.5*xp
    yp_actual=np.array(
        [s1(x) for x in xp]
    )

    plt.figure()
    plt.plot(xd,yd,"xr",label="data")
    plt.plot(xp, yp_actual, "--k", label="s1")
    eps_t=np.linalg.norm(yp_exp-yp_actual)/np.linalg.norm(yp_exp)
    plt.text(0.5,-1.0,f"eps_t={eps_t}")
    plt.legend()
    plt.savefig("figures/test_linear.png")


    xd=np.linspace(-5.0,5.0,10)
    yd=1.0+0.5*xd+0.25*xd**2

    s2=spline_function(xd,yd,order=2)
    xp=np.linspace(-6.0,6.0,100)
    yp_exp=1.0+0.5*xp+0.25*xp**2
    yp_actual=np.array(
        [s2(x) for x in xp]
    )

    plt.figure()
    plt.plot(xd,yd,"xr",label="data")
    plt.plot(xp, yp_actual, "--k", label="s2")
    eps_t=np.linalg.norm(yp_exp-yp_actual)/np.linalg.norm(yp_exp)
    plt.text(0.5,-1.0,f"eps_t={eps_t}")
    plt.legend()
    plt.savefig("figures/test_quadratic.png")

if __name__=="__main__":
    test_gauss_iter_solve()
    test_spline()


import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.insert(1, '/goph419-f2023-lab02-stSL/src/lab02')

from linalg_interp import (
    gauss_iter_solve,
    spline_function,
)

def main():
    water_temps=np.loadtxt("water_density_vs_temp_usgs.txt", dtype=float)

    water_t=water_temps[:,0]
    water_d=water_temps[:,1]

    water_x=np.linspace(np.min(water_t),np.max(water_t),100)

    s1=spline_function(water_t,water_d, order=1)
    s2=spline_function(water_t,water_d, order=2)
    s3=spline_function(water_t,water_d, order=3)

    water_y1=np.array([s1(x) for x in water_x])
    water_y2=np.array([s2(x) for x in water_x])
    water_y3=np.array([s3(x) for x in water_x])

    fig,ax=plt.subplots(2,3, sharex="row", figsize=((13,13)))

    ax[0][0].plot(water_x, water_y1, 'y--', label="linear spline case")
    ax[0][0].plot(water_t, water_d, 'xr', label='data')
    ax[0][0].set(xlabel="Temperature (C)", ylabel="Density (g/cm^3)",
        title="Water Density vs. Temperature (Linear)"
               )
    ax[0][0].legend()
    
    ax[0][1].plot(water_x, water_y2, 'y--', label="quadratic spline case")
    ax[0][1].plot(water_t, water_d, 'xr', label='data')
    ax[0][1].set(xlabel="Temperature (C)", ylabel="Density (g/cm^3)",
        title="Water Density vs. Temperature (Quadratic)"
               )
    ax[0][1].legend()

    ax[0][2].plot(water_x, water_y3, 'y--', label="cubic spline case")
    ax[0][2].plot(water_t, water_d, 'xr', label='data')
    ax[0][2].set(xlabel="Temperature (C)", ylabel="Density (g/cm^3)",
        title="Water Density vs. Temperature (Cubic)"
               )
    ax[0][2].legend()

    air_temps=np.loadtxt("air_density_vs_temp_eng_toolbox.txt", dtype=float)

    air_t=air_temps[:,0]
    air_d=air_temps[:,1]

    air_x=np.linspace(np.min(air_t),np.max(air_t),100)

    s1=spline_function(air_t,air_d, order=1)
    s2=spline_function(air_t,air_d, order=2)
    s3=spline_function(air_t,air_d, order=3)

    air_y1=np.array([s1(x) for x in air_x])
    air_y2=np.array([s2(x) for x in air_x])
    air_y3=np.array([s3(x) for x in air_x])

    ax[1][0].plot(air_x, air_y1, 'y--', label="linear spline case")
    ax[1][0].plot(air_t, air_d, 'xr', label='data')
    ax[1][0].set(xlabel="Temperature (C)", ylabel="Density(kg/m^3)",
        title="Air Density vs. Temperature (Linear)"
               )
    ax[1][0].legend()
    
    ax[1][1].plot(air_x, air_y2, 'y--', label="quadratic spline case")
    ax[1][1].plot(air_t, air_d, 'xr', label='data')
    ax[1][1].set(xlabel="Temperature (C)", ylabel="Density(kg/m^3)",
        title="Air Density vs. Temperature (Quadratic)"
               )
    ax[1][1].legend()

    ax[1][2].plot(air_x, air_y3, 'y--', label="cubic spline case")
    ax[1][2].plot(air_t, air_d, 'xr', label='data')
    ax[1][2].set(xlabel="Temperature (C)", ylabel="Density(kg/m^3)",
        title="Air Density vs. Temperature (Cubic)"
               )
    ax[1][2].legend()

    fig.savefig("../figures/densities_temp.png")
    plt.show()

if __name__=="__main__":
    main()
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

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

    spline1=spline_function(water_t,water_d, order=1)
    spline2=spline_function(water_t,water_d, order=2)
    spline3=CubicSpline(water_t,water_d)

    water_y1=spline1(water_x)
    water_y2=spline2(water_x)
    water_y3=spline3(water_x)

    fig,axis=plt.subplots(nrows=2,ncolumns=3, figsize=(15,15))

    axis[0][0].plot(water_x, water_y1, 'y--', label="linear spline case")
    axis[0][0].set_xlabel("Temperature(C)")
    

    air_temps=np.loadtxt("air_density_vs_temp_eng_toolbox.txt", dtype=float)

    air_t=air_temps[:,0]
    air_d=air_temps[:,1]



if __name__=="__main__":
    main()
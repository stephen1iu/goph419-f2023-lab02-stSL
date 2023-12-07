import numpy as np
import sys

sys.path.insert(1, '/goph419-f2023-lab02-stSL/src/lab02')

from linalg_interp import (
    gauss_iter_solve,
    spline_function,
)

def main():
    alg=(input(""))
    water_temps=np.loadtext("water_density_vs_temp_usgs.txt")
    air_temps=np.loadtext("air_density_vs_temp_eng_toolbox.txt")


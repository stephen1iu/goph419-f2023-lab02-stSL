import numpy as np

def main():
    alg=(input(""))
    water_temps=np.loadtext("water_density_vs_temp_usgs.txt")
    air_temps=np.loadtext("air_density_vs_temp_eng_toolbox.txt")
    
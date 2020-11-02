import pandas as pd
import numpy as np

# input file names
INPUT_FILE_1 = "bond1.dat"
INPUT_FILE_2 = "bond2.dat"
INPUT_FILE_3 = "angle.dat"

#reading in dataframes
df1 = pd.read_table("bond1.dat", header=None)
df2 = pd.read_table("bond2.dat", header=None) 
df3 = pd.read_table("angle.dat", header=None)

#merging data frames
tmp = df1.merge(df2, on=0)
geom = tmp.merge(df3, on=0)

#column names
geom.columns = ["CONF", "BOND1", "BOND2", "ANGLE"]

#print check
print(geom)

#formatting and saving output to data file
np.savetxt("h2o_geom.dat", geom.values, fmt=['%i', '%3.6f', '%3.6f', '%3.6f'], delimiter="\t")


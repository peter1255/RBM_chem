import pandas as pd
import numpy as np

FILE_PATH = "ethane_rbm_output.txt"
OUT_FILE = "ethane_rbm_xyz.dat"
NUM_CONFS = 10000
NUM_PARAMS = 18

rbm_out = np.getfromtxt(FILE_PATH)

with open(OUT_FILE, "a") as f:
    for i in range(NUM_CONFS):
        f.write("\nC")
        f.write("\nC\t1\t{}".format(rbm_input[i, 0]))
        f.write("\nH\t1\t{}\t2\t{}".format(rbm_input[i, 1], rbm_input[i, 2]))
        f.write("\nH\t1\t{}\t2\t{}\t3\t{}".format(rbm_input[i, 3], rbm_input[i, 4], rbm_input[i, 5]))

        rbm_input[i, 1] = zmat.iloc[i*8+2,2]
        rbm_input[i, 2] = zmat.iloc[i*8+2,4]
        rbm_input[i, 3] = zmat.iloc[i*8+3,2]
        rbm_input[i, 4] = zmat.iloc[i*8+3,4]
        rbm_input[i, 5] = zmat.iloc[i*8+3,6]
        rbm_input[i, 6] = zmat.iloc[i*8+4,2]
        rbm_input[i, 7] = zmat.iloc[i*8+4,4]
        rbm_input[i, 8] = zmat.iloc[i*8+4,6]
        rbm_input[i, 9] = zmat.iloc[i*8+5,2]
        rbm_input[i, 10] = zmat.iloc[i*8+5,4]
        rbm_input[i, 11] = zmat.iloc[i*8+5,6]
        rbm_input[i, 12] = zmat.iloc[i*8+6,2]
        rbm_input[i, 13] = zmat.iloc[i*8+6,4]
        rbm_input[i, 14] = zmat.iloc[i*8+6,6]
        rbm_input[i, 15] = zmat.iloc[i*8+7,2]
        rbm_input[i, 16] = zmat.iloc[i*8+7,4]
        rbm_input[i, 17] = zmat.iloc[i*8+7,6]

np.savetxt(OUT_FILE, rbm_input)

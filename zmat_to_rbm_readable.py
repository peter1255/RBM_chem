import pandas as pd
import numpy as np

FILE_PATH = "zmatrix.dat"
OUT_FILE = "ethane_rbm_input_2.dat"
NUM_CONFS = 10000
NUM_PARAMS = 18

my_cols = [*range(7)]
zmat = pd.read_csv(FILE_PATH, names=my_cols, delim_whitespace=True, header=None)

test = zmat.iloc[5,4]
rbm_input = np.zeros([NUM_CONFS, NUM_PARAMS])

for i in range(NUM_CONFS):
    rbm_input[i, 0] = zmat.iloc[i*8+1,2]
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

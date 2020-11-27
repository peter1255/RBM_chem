import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tfrbm import BBRBM, GBRBM, GDBM, GGRBM, ReluRBM, TanhRBM, LeakyReluRBM
from matplotlib.colors import Normalize
import tensorflow as tf

confs = np.genfromtxt("h2o_scf.txt")
gbrbm = GBRBM(n_visible=3, n_hidden=12, learning_rate=0.001, sigma=0.7, err_function='mse')
gbrbm_err, _ = gbrbm.fit(confs, n_epoches=20, batch_size=20)

fig1, ax1 = plt.subplots()
ax1.plot(gbrbm_err)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('MSE')
plt.show()

sims = gbrbm.simulate("simulated.out", confs=10000, postprocess=True)


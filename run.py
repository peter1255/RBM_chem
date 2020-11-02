import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tfrbm import BBRBM, GBRBM, GDBM, GGRBM, ReluRBM, TanhRBM, LeakyReluRBM
from matplotlib.colors import Normalize
import tensorflow as tf

confs = np.genfromtxt("cross_2D_laplace.txt")
gbrbm = GBRBM(n_visible=2, n_hidden=4, learning_rate=0.001, sigma=0.7, err_function='mse')
gbrbm_err, gbrbm_mle = gbrbm.fit(confs, n_epoches=20, batch_size=20)

fig1, ax1 = plt.subplots()
ax1.plot(gbrbm_mle)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Log-Likelihood')
plt.show()

ggrbm = GGRBM(n_visible=2, n_hidden=4, learning_rate=0.001, sigma=0.7, err_function='mse')
relu = ReluRBM(n_visible=2, n_hidden=4, learning_rate=0.001, sigma=0.7, err_function='mse')
leaky = LeakyReluRBM(n_visible=2, n_hidden=4, learning_rate=0.001, sigma=0.7, err_function='mse')
tanh = TanhRBM(n_visible=2, n_hidden=4, learning_rate=0.0001, sigma=0.7, err_function='mse')
confs = gbrbm.preprocess_data(confs)
ggrbm_err = ggrbm.fit(confs, n_epoches=7, batch_size=20)
relu_err = relu.fit(confs, n_epoches=7, batch_size=20)
leaky_err = leaky.fit(confs, n_epoches=7, batch_size=20)
tanh_err = tanh.fit(confs, n_epoches=7, batch_size=20)

gbrbm_err = np.mean(gbrbm_err.reshape(-1, 50), axis=1)
ggrbm_err = np.mean(ggrbm_err.reshape(-1, 50), axis=1)
relu_err = np.mean(relu_err.reshape(-1, 50), axis=1)
leaky_err = np.mean(leaky_err.reshape(-1, 50), axis=1)
tanh_err = np.mean(tanh_err.reshape(-1, 50), axis=1)

epochs = np.linspace(0, gbrbm.n_epoches, num=gbrbm_err.shape[0])


#hidden_nodes_before_training = rbm.transform(confs)
#rbm.load_weights("ethane")
#rbm.save_weights("ethane_2", overwrite=True)
#hidden_nodes_after_training = rbm.transform(confs)
#sims = rbm.simulate("ethane_rbm_output.txt", confs=10000, delta=1.0, postprocess=True)
#st = np.std(sims, axis=0)


'''
fig1, axs1 = plt.subplots(1, 2, sharey=True, sharex=True, tight_layout=True)
before_avg = np.average(hidden_nodes_before_training, axis=0)
after_avg = np.average(hidden_nodes_after_training, axis=0)
axs1[0].hist(before_avg, bins=30)
axs1[1].hist(after_avg, bins=30)


visibles = np.zeros([rbm.n_hidden, rbm.n_visible])
for i in range(rbm.n_hidden):
    hidden_layer = np.zeros((1,rbm.n_hidden))
    hidden_layer[0,i] = 1
    visibles[i] = rbm.transform_inv(hidden_layer)

#print(visibles)


visibles = gbrbm.postprocess_data(visibles)
f=open('hidden_filters_3D_laplace.txt','ab')
np.savetxt(f, visibles, fmt='%4.8f')
f.close()

jet=plt.get_cmap('coolwarm')


fig2, ax2 = plt.subplots()
ax2.scatter(sims[:,0], sims[:,1], alpha=0.50, s=0.50)
plt.show()


weights = np.genfromtxt("hidden_filters.txt")
dims=2
q1=np.zeros(dims)
q2=np.zeros(dims)
q3=np.zeros(dims)
q4=np.zeros(dims)

for x in weights:
    if x[0] > 0 and x[1] > 0:
        q1+=x
    elif x[0] < 0 and x[1] > 0:
        q2+=x
    elif x[0] < 0 and x[1] < 0:
        q3+=x
    elif x[0] > 0 and x[1] < 0:
        q4+=x

#weights = np.array([q1/100,q2/100,q3/100,q4/100])


#offset=1/np.std(sims, 0)


x = np.arange(-3, 3, 0.05) #np.arange(-8.6, 15.2, 0.1) #x = np.arange(-4, 6, 0.1)
y = np.arange(-3, 3, 0.05) #np.arange(-8.6, 15.2, 0.1) #y = np.arange(-4, 6, 0.1)
x, y = np.meshgrid(x, y)
z = rbm.contour(x, y)

#x = x*rbm.stdevs[0]+rbm.means[0]
#y = y*rbm.stdevs[1]+rbm.means[1]
#np.savetxt("gbrbm_contour_h2o_ba_x.txt",x)
#np.savetxt("gbrbm_contour_h2o_ba_y.txt",y)
#np.savetxt("gbrbm_contour_h2o_ba_z.txt",z)


#weights = gdbm.w.numpy()

origin=[0,0]
fig3, ax3 = plt.subplots()
ax3.quiver(origin[0], origin[1], visibles[:,0], visibles[:,1], scale=5) #color=cm(norm(visibles[:,gbrbm.n_visible-1])),
ax3.contour(x,y,z, levels=25, cmap="viridis")

plt.show()




x = np.arange(min[2]-3, max[2]+2, 0.1)
y = gbrbm.energy_dist(x)
x = x*gbrbm.stdevs[2]+gbrbm.means[2]

np.savetxt("gbrbm_h2o_angle2.txt", np.column_stack((x,y)))



fig4, ax4 = plt.subplots()
ax4.plot(x, y)

plt.show()

'''

fig5, ax5 = plt.subplots()
ax5.plot(epochs, gbrbm_err, label="GBRBM")
ax5.plot(epochs, ggrbm_err, label="GGRBM")
ax5.plot(epochs, relu_err, label="ReLU RBM")
ax5.plot(epochs, leaky_err, label="Leaky ReLU RBM")
ax5.plot(epochs, tanh_err, label="Tanh RBM")

ax5.set_xlabel('Epochs')
ax5.set_ylabel('Reconstruction Error (MSE)')
plt.legend()
plt.show()


'''
plt.title('Deep BM')
ax5[1].plot(errs[1])
ax5[1].set_xlabel('batches')
ax5[1].set_ylabel(rbm.err_function)
secax = ax5[1].secondary_xaxis('top', functions=(lambda a: a / rbm.n_batches, lambda a: a * rbm.n_batches))
secax.set_xlabel("epochs")
'''

#bbrbm = BBRBM(n_visible=90, n_hidden=50, learning_rate=0.01, momentum=0.95, use_tqdm=True, err_function='mse')
#print(np.cov(np.transpose(hidden_nodes_after_training)))

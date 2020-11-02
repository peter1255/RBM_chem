import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.preprocessing import normalize

ORIGINAL_DISTRIBUTION = "h2o_scf.dat" # "h2o_geom_from_eq.dat"
SIMULATED_DISTRIBUTION = "gbrbm_mc_output_h2o_scf.txt" # "gbrbm_mc_output_different_distribution_sigmas.txt"

original_distribution = np.loadtxt(ORIGINAL_DISTRIBUTION)
simulated_distribution = np.loadtxt(SIMULATED_DISTRIBUTION)
titles = ["BOND1", "BOND2", "ANGLE"]

fig, axs = plt.subplots(2,3)

for i in range(3):
    axs[0,i].set_title(titles[i])
    og_range = (original_distribution[:,i].min(), original_distribution[:,i].max())
    sim_range = (simulated_distribution[:,i].min(), simulated_distribution[:,i].max())
    if og_range[1] > sim_range[1]:
        ran = og_range
    else:
        ran = sim_range
    axs[0,i].hist(original_distribution[:,i], density=True, color='blue', edgecolor='white', bins=25,
                  range=ran, label="Mean: {:.6f} \n StDev: {:.6f}".format(np.mean(original_distribution[:,i]), np.std(original_distribution[:,i])))
    axs[0,i].legend()
    bottom, top = axs[0,i].get_ylim()
    axs[1,i].hist(simulated_distribution[:,i], density=True, color='green', edgecolor='white', bins=25,
                  range=ran, label="Mean: {:.6f} \n StDev: {:.6f}".format(np.mean(simulated_distribution[:,i]), np.std(simulated_distribution[:,i])))
    axs[1,i].legend()
    axs[1,i].set_ylim((bottom, top))

fig, axs = plt.subplots(2,3)

for i in range(3):
    axs[0,i].set_title("{} vs. {}".format(titles[i], titles[(i+1)%3]))
    axs[0,i].scatter(original_distribution[:, i], original_distribution[:, (i+1)%3], alpha=0.50, s=0.50)
    axs[1,i].scatter(simulated_distribution[:, i], simulated_distribution[:, (i+1)%3], c='green', alpha=0.50, s=0.50)
    plt.setp(axs[1,i], ylim=axs[0, i].get_ylim(), xlim=axs[0,i].get_xlim())

plt.setp(axs[0,0], ylabel="ORIGINAL")
plt.setp(axs[1,0], ylabel="SIMULATED")

fig = plt.figure()
jet=plt.get_cmap('coolwarm')

#visibles = np.loadtxt("hidden_filters.txt")
#norm = Normalize()
#norm.autoscale(visibles[:,2])
#cm = cm.coolwarm
#sm = cm.ScalarMappable(cmap=cm, norm=norm)
#sm.set_array([])
#origin = np.array([0.96070164, 0.95904821, 104.54626591])

ax1 = fig.add_subplot(121, xlabel="ORIGINAL")
ax1.scatter(original_distribution[:, 0], original_distribution[:,1], c=original_distribution[:,2], cmap=jet, vmin=-3, vmax=3, alpha=0.5, s=0.50)
ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1, xlabel="SIMULATED")
hi = ax2.scatter(simulated_distribution[:, 0], simulated_distribution[:,1], c=simulated_distribution[:,2], cmap=jet, vmin=-3, vmax=3, alpha=0.5, s=0.50)
#ax2.quiver(origin[0], origin[1], visibles[:,0], visibles[:,1], color=cm(norm(visibles[:,2])), alpha=0.8)

cbax = fig.add_axes([0.93, 0.115, 0.02, 0.77])
fig.colorbar(hi, cax=cbax)

plt.show()
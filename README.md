# RESTRICTED BOLTZMAN MACHINE LEARNING FOR APPLICATIONS IN COMPUTATIONAL CHEMISTRY 

## PREREQUISITES 

- Python 3.7
- TensorFlow 1.5 or higher
- NumPy 1.11 or higher
- Pandas 0.25 or higher

## PREPROCESSING DATA 

This program will require an input file which contains information about the system (i.e. bond lengths, angles, torsions) separated by column. Information can be either binary or continuous. See h2o_geom.dat for an example. If you're starting with a raw coordinate file (.xyz), there is a script in the directory (calculate_bonds.py) that allows you to calculate bond lengths and angles, but will likely need to be modified for a different molecule. Once you have the data properly formatted we load it into the program using Numpy:

```
confs = np.genfromtxt("h2o_scf.txt")
```

If you have continuous-valued data you'll want to shift and scale the data to mean 0 and unit variance. This is done using:

```
confs = tanh.preprocess_data(confs)
```

The RBM stores the mean and variances of each parameter so that after a simulation is performed at mean 0 and unit variance, the data is then post-processed (shifted and scaled) to match the mean and variance of the original data. 

# CHOOSING A MODEL

Different types of RBMs are available in this package. If you have binary data you must use a binary-binary RBM (BBRBM), but otherwise you are free to use Gaussian-Binary RBM, Tanh RBM, ReLU RBM, etc. which differ in the activation function for the hidden layer. Different activation functions may work better on different types of data. It is generally recommended to use Gaussian-Binary RBM (GBRBM) since it is the most versatile and easy to use. In order to instantiate the model you must create a RBM object with a specified number of visible and hidden nodes. You can also specify the learning rate and the error function used by the RBM:

```
gbrbm = GBRBM(n_visible=3, n_hidden=12, learning_rate=0.001, err_function='mse')
```

In order to train the model you must use the "fit" function found within the RBM subclass:

```
err, LL = gbrbm.fit(confs, n_epoches=20, batch_size=20, LL=True)
```

This step will return both the error during training and the log-likelihood estimate, provided we set the log-likehood argument 'LL=True'. We can then plot the error/log-likelihood during training using matplotlib:

```
fig1, ax1 = plt.subplots()
ax1.plot(gbrbm_err)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('MSE')
plt.show()
```

# SIMULATION

In order to generate new samples, we use the simulate function:

```
sims = rbm.simulate("ethane_rbm_output.txt", confs=10000, postprocess=True)
```

# ACCURACY OF MODEL

AUTHOR: Peter Nekrasov

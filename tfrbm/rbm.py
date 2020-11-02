from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import sys
from .util import tf_xavier_init, sample_bernoulli, sample_gaussian, kbt
import random
import math


class RBM:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 learning_rate=0.01,
                 xavier_const=1.0,
                 err_function='mse',
                 use_tqdm=True):

        if err_function not in {'mse', 'cosine', 'compare', 'compare_available', 'compare_missing'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\' or \'compare\'')

        self._use_tqdm = use_tqdm

        if use_tqdm is not None:
            from tqdm import tqdm
            self._tqdm = tqdm

        self.err_function = err_function
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate

        self.x = None
        self.y = None

        self.w = tf.random.normal([self.n_visible, self.n_hidden], mean=0, stddev=0.1)
        self.visible_bias = tf.zeros([self.n_visible], dtype=tf.float32)
        self.hidden_bias = tf.zeros([self.n_hidden], dtype=tf.float32)

        self.delta_w = tf.zeros([self.n_visible, self.n_hidden], dtype=tf.float32)
        self.delta_visible_bias = tf.zeros([self.n_visible], dtype=tf.float32)
        self.delta_hidden_bias = tf.zeros([self.n_hidden], dtype=tf.float32)

        self.compute_hidden = None
        self.compute_visible = None
        self.n_batches = None
        self.n_epoches = None
        self.test = None
        self.batch_size = None
        self.train = None

    def fit(self,
            data_x, c=0,
            n_epoches=10,
            batch_size=10,
            shuffle=True,
            verbose=True):

        assert n_epoches > 0

        self.c = c
        self.test = False
        self.batch_size = batch_size
        self.n_epoches = n_epoches
        n_data = data_x.shape[0]

        if batch_size > 0:
            self.n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            self.n_batches = 1

        if shuffle:
            data_x_cpy = data_x.copy()
            inds = np.arange(n_data)
        else:
            data_x_cpy = data_x

        errs = []
        LLs = np.zeros(n_epoches)

        for e in range(n_epoches):

            if verbose and not self._use_tqdm:
                print('Epoch: {:d}'.format(e))

            epoch_errs = np.zeros((self.n_batches,))
            epoch_errs_per_param = np.zeros((n_data, self.n_visible))
            epoch_errs_ptr = 0

            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]

            r_batches = range(self.n_batches)

            if verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(e), ascii=True, file=sys.stdout)

            for b in r_batches:
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]
                self.x = tf.Variable(batch_x, dtype=tf.float32)

                self._initialize_vars()

                batch_err, batch_errs_per_param = self.get_err(self.x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_per_param[epoch_errs_ptr*batch_size:(epoch_errs_ptr+1)*batch_size] = batch_errs_per_param
                epoch_errs_ptr += 1

            if verbose:
                err_mean = epoch_errs.mean()
                if self._use_tqdm:
                    self._tqdm.write('Train error: {:.4f}'.format(err_mean))
                    self._tqdm.write('')
                else:
                    print('Train error: {:.4f}'.format(err_mean))
                    print('')
                sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])
            LLs[e] = self.get_log_likelihood(tf.convert_to_tensor(data_x, dtype=tf.float32))

        return errs, LLs

    def train_and_test(self,
                       data_x, test_data, key,
                       n_epoches=10,
                       batch_size=10,
                       shuffle=True,
                       verbose=True):

        assert n_epoches > 0

        self.batch_size = batch_size
        n_data = data_x.shape[0]

        if batch_size > 0:
            self.n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            self.n_batches = 1

        if shuffle:
            data_x_cpy = data_x.copy()
            inds = np.arange(n_data)
        else:
            data_x_cpy = data_x

        test_errs = np.zeros((n_epoches,))
        train_errs = np.zeros((n_epoches,))

        errs = []

        for e in range(n_epoches):

            if verbose and not self._use_tqdm:
                print('Epoch: {:d}'.format(e))

            epoch_errs = np.zeros((self.n_batches,))
            epoch_errs_ptr = 0

            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]

            r_batches = range(self.n_batches)

            if verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(e), ascii=True, file=sys.stdout)

            self.train = True



            for b in r_batches:
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]
                self.x = tf.Variable(batch_x, dtype=tf.float32)

                self._initialize_vars()

                #self.partial_fit(batch_x)
                batch_err, batch_err_per_param = self.get_err(batch_x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1


            # performing predictions on the test data
            self.train = False
            self.x = tf.Variable(test_data, dtype=tf.float32)
            self._initialize_vars()

            test_err = self.get_err(key)
            test_errs[e] = test_err

            err_mean = epoch_errs.mean()
            train_errs[e] = err_mean

            if verbose:
                if self._use_tqdm:
                    self._tqdm.write('Train acc: {:.4f}, Test acc: {:.4f}'.format(err_mean, test_err))
                    self._tqdm.write('')
                else:
                    print('Train error: {:.4f}'.format(err_mean))
                    print('')
                sys.stdout.flush()

        return train_errs, test_errs

    def _initialize_vars(self):
        pass

    def get_err(self, batch_x):
        error_per_node=None

        if self.err_function == 'cosine':
            x1_norm = tf.nn.l2_normalize(batch_x, 1)
            x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
            cos_val = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm), 1))
            self.compute_err = tf.acos(cos_val) / tf.constant(np.pi)

        if self.err_function == 'mse':
            square_error = tf.square(batch_x - self.compute_visible)
            self.compute_err = tf.reduce_mean(square_error)

        if self.err_function == 'compare':
            x = tf.reduce_all(tf.equal(self.compute_visible, batch_x), axis=1)
            self.compute_err = tf.reduce_sum(tf.cast(x, tf.float32)) / tf.cast(tf.shape(batch_x)[0], tf.float32)

        return self.compute_err, error_per_node

    def get_energy(self):
        pass

    def keep_or_not(self, prev_energy):
        current_energy = self.get_energy()
        if current_energy <= prev_energy:
            return True
        probs = tf.math.exp((prev_energy - current_energy) / 1)
        if random.random() < probs:
            return True
        else:
            return False

    def simulate(self,
                 filename="simulated_confs.txt",
                 confs=1000,
                 delta=1.0,
                 postprocess=True):
        self.x = tf.zeros([1, self.n_visible])
        self._compute_hidden()
        conf_table = np.zeros([confs, self.n_visible])
        accepted_confs=0
        confs_rejected=0
        pbar = self._tqdm(total=confs, file=sys.stdout)
        while accepted_confs < confs:
            prev_energy = self.get_energy()
            orig_x = tf.identity(self.x)
            orig_h = tf.identity(self.compute_hidden)
            self.x = self.x + tf.random.normal([1, self.n_visible], stddev=delta)
            self._compute_hidden()
            if self.keep_or_not(prev_energy):
                conf_table[accepted_confs,:] = self.x.numpy()
                pbar.update(1)
                accepted_confs+=1
            else:
                self.x = orig_x
                self.compute_hidden = orig_h
                confs_rejected+=1

        if postprocess:
            conf_table = self.postprocess_data(conf_table)
    
        np.savetxt(filename, conf_table, fmt='%.18f')
        pbar.close()

        percent_accepted = accepted_confs / (accepted_confs + confs_rejected)
        print("PERCENT CONFS ACCEPTED: {:.1f}%".format(percent_accepted*100))

        return conf_table

    def AIS(self):
        def f_0(x):
            # Target distribution: proportional to energy function
            return tf.math.exp(-self.get_energy(x))

        def f_n(x):
            # Proposal distribution: similar to the target distribution but has known partition function
            return tf.math.exp(-(tf.reduce_sum(tf.square(x) / (2 * np.square(self.sigma)))))

        def f_j(x, beta):
            # Intermediate distribution: interpolation between f_0 and f_n
            return f_0(x).numpy() ** beta * f_n(x).numpy() ** (1 - beta)

        def T(x, beta, n_steps=10):
            """
            Transition distribution: T(x'|x) using n-steps Metropolis sampler
            """
            for t in range(n_steps):
                # Proposal
                x_prime = x + tf.random.normal([1, self.n_visible], stddev=1)

                # Acceptance prob
                a = f_j(x_prime, beta) / f_j(x, beta)

                if np.random.rand() < a:
                    x = x_prime

            return x

        n_inter = 50  # num of intermediate dists
        betas = np.linspace(0, 1, n_inter)

        # Sampling
        n_samples = 100
        samples = np.zeros(n_samples)
        weights = np.zeros(n_samples)

        for t in range(n_samples):
            # Sample initial point from q(x)
            x = tf.random.normal([1,self.n_visible], mean=0, stddev=tf.math.sqrt(1/2))
            w = 1

            for n in range(1, len(betas)):
                # Transition
                x = T(x, betas[n], n_steps=5)

                # Compute weight in log space (log-sum):
                # w *= f_{n-1}(x_{n-1}) / f_n(x_{n-1})
                w += np.log(f_j(x, betas[n])) - np.log(f_j(x, betas[n - 1]))

            weights[t] = np.exp(w)  # Transform back using exp

        # Compute expectation
        z = np.sum(weights)
        return z

    def get_log_likelihood(self, batch_x):
        Z = 1 #self.AIS()
        self.x = batch_x
        LL = - tf.reduce_sum(self.get_energy()) - np.log(Z)
        return LL


    def transform(self, batch_x):
        self.x = tf.Variable(batch_x, dtype=tf.float32)
        self._compute_hidden()
        return self.compute_hidden.numpy()

    def contour(self, x, y):
        z = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                self.x = tf.constant([[x[i, j], y[i, j], 0]], dtype=tf.float32)
                self._compute_hidden()
                z[i, j] = self.get_energy()
            print(i)
        return z

    def energy_dist(self, x):
        z = np.zeros(x.shape)
        for i in range(x.shape[0]):
            self.x = tf.constant([[0, 0, x[i]]], dtype=tf.float32)
            self._compute_hidden()
            z[i] = self.get_energy()
        return z

    def transform_inv(self, batch_y):
        self.compute_hidden = tf.Variable(batch_y, dtype=tf.float32) # is tf.Variable necessary here ?
        self._compute_visible()
        return self.compute_visible.numpy()

    def reconstruct(self, batch_x):
        self.x = batch_x
        self._compute_hidden()
        self._compute_visible()
        err = self.get_err(batch_x)
        return self.compute_visible, err

    def get_weights(self):
        return self.w.numpy(), self.visible_bias.numpy(), self.hidden_bias.numpy()

    def save_weights(self, prefix, overwrite=False):
        if overwrite:
            mode = 'w+'
        else:
            mode = 'ab'
        f=open("{}_weights.txt".format(prefix),mode)
        g=open("{}_visible_bias.txt".format(prefix),mode)
        h=open("{}_hidden_bias.txt".format(prefix),mode)
        np.savetxt(f, self.w.numpy())
        np.savetxt(g, self.visible_bias.numpy())
        np.savetxt(h, self.hidden_bias.numpy())
        f.close()
        g.close()
        h.close()

    def load_weights(self, prefix):
        weights = np.loadtxt("{}_weights.txt".format(prefix))
        visible_bias = np.loadtxt("{}_visible_bias.txt".format(prefix))
        hidden_bias = np.loadtxt("{}_hidden_bias.txt".format(prefix))
        self.w = tf.Variable(weights, dtype=tf.float32)
        self.visible_bias = tf.Variable(visible_bias, dtype=tf.float32)
        self.hidden_bias = tf.Variable(hidden_bias, dtype=tf.float32)

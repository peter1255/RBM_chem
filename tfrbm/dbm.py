from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import sys
from .util import tf_xavier_init, sample_bernoulli, sample_gaussian, kbt
import random
import math


class DBM:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 n_hidden_2,
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
        self.n_hidden_2 = n_hidden_2
        self.learning_rate = learning_rate

        self.x = None
        self.y = None
        self.r1 = None
        self.r2 = None

        self.w = tf.random.normal([self.n_visible, self.n_hidden], mean=0, stddev=0.01)  #
        self.w_2 = tf.random.normal([self.n_hidden, self.n_hidden_2], mean=0, stddev=0.01) #
        self.visible_bias = tf.zeros([self.n_visible], dtype=tf.float32)  #
        #self.hidden_bias = tf.zeros([self.n_hidden], dtype=tf.float32)  #
        self.hidden_bias = tf.Variable([-2., -2., -2., -2.])
        self.hidden_bias_2 = tf.zeros([self.n_hidden_2], dtype=tf.float32) #

        self.compute_hidden = None
        self.compute_visible = None
        self.n_batches = None
        self.test = None
        self.batch_size = None
        self.train = None
        self.sigma = None
        self.mf_steps = 5
        self.n_particles = 100

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

        self.print_param()

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

                self.pretrain_first_layer()

                batch_err, batch_errs_per_param = self.get_err(self.x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_per_param[
                epoch_errs_ptr * batch_size:(epoch_errs_ptr + 1) * batch_size] = batch_errs_per_param
                epoch_errs_ptr += 1

            if verbose:
                err_mean = epoch_errs.mean()
                if self._use_tqdm:
                    self._tqdm.write('Pretrain layer 1 error: {:.4f}'.format(err_mean))
                    self._tqdm.write('')
                else:
                    print('Pretrain layer 1 error: {:.4f}'.format(err_mean))
                    print('')
                sys.stdout.flush()

            errs_1 = np.hstack([errs, epoch_errs])

        self.print_param()

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

                self.pretrain_second_layer()

                batch_err, batch_errs_per_param = self.get_err(self.x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_per_param[
                epoch_errs_ptr * batch_size:(epoch_errs_ptr + 1) * batch_size] = batch_errs_per_param
                epoch_errs_ptr += 1

            if verbose:
                err_mean = epoch_errs.mean()
                if self._use_tqdm:
                    self._tqdm.write('Pretrain layer 2 error: {:.4f}'.format(err_mean))
                    self._tqdm.write('')
                else:
                    print('Pretrain layer 2 error: {:.4f}'.format(err_mean))
                    print('')
                sys.stdout.flush()

            errs_2 = np.hstack([errs, epoch_errs])

        self.print_param()

        self.r1 = tf.identity(self.w)
        self.r2 = tf.identity(self.w_2)

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

            self.v_tilde = tf.random.normal([self.n_particles, self.n_visible])
            self.h_1_tilde = tf.random.normal([self.n_particles, self.n_hidden])
            self.h_2_tilde = tf.random.normal([self.n_particles, self.n_hidden_2])

            for b in r_batches:
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]
                self.x = tf.Variable(batch_x, dtype=tf.float32)

                self.global_optimization()

                batch_err, batch_errs_per_param = self.get_err(self.x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_per_param[
                epoch_errs_ptr * batch_size:(epoch_errs_ptr + 1) * batch_size] = batch_errs_per_param
                epoch_errs_ptr += 1

            if verbose:
                err_mean = epoch_errs.mean()
                if self._use_tqdm:
                    self._tqdm.write('Global optimization error: {:.4f}'.format(err_mean))
                    self._tqdm.write('')
                else:
                    print('Global optimization error: {:.4f}'.format(err_mean))
                    print('')
                sys.stdout.flush()

        self.print_param()

        return errs_1, errs_2

    def print_param(self):
        print(self.w)
        print(self.w_2)
        print(self.visible_bias)
        print(self.hidden_bias)
        print(self.hidden_bias_2)

    def _initialize_vars(self):
        pass

    def get_err(self, batch_x):
        error_per_node = None

        if self.err_function == 'cosine':
            x1_norm = tf.nn.l2_normalize(batch_x, 1)
            x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
            cos_val = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm), 1))
            self.compute_err = tf.acos(cos_val) / tf.constant(np.pi)

        if self.err_function == 'mse':
            square_error = tf.square(batch_x - self.compute_visible)
            error_per_node = self.compute_visible
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
        self.inference()
        conf_table = np.zeros([confs, self.n_visible])
        accepted_confs = 0
        confs_rejected = 0
        pbar = self._tqdm(total=confs, file=sys.stdout)
        while accepted_confs < confs:
            prev_energy = self.get_energy()
            orig_x = tf.identity(self.x)
            orig_h = tf.identity(self.compute_hidden)
            orig_h_2 = tf.identity(self.compute_hidden_2)
            self.x = self.x + tf.random.normal([1, self.n_visible], stddev=delta)
            self.inference()
            if self.keep_or_not(prev_energy):
                conf_table[accepted_confs, :] = self.x.numpy()
                pbar.update(1)
                accepted_confs += 1
            else:
                self.x = orig_x
                self.compute_hidden = orig_h
                self.compute_hidden_2 = orig_h_2
                confs_rejected += 1

        if postprocess:
            conf_table = self.postprocess_data(conf_table)

        np.savetxt(filename, conf_table, fmt='%.18f')
        pbar.close()

        percent_accepted = accepted_confs / (accepted_confs + confs_rejected)
        print("PERCENT CONFS ACCEPTED: {:.1f}%".format(percent_accepted * 100))

        return conf_table

    def transform(self, batch_x):
        self.x = tf.Variable(batch_x, dtype=tf.float32)
        self.compute_hidden = sample_bernoulli(tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias))
        return self.compute_hidden.numpy()

    def contour(self, x, y):
        z = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                self.x = tf.constant([[x[i, j], y[i, j]]], dtype=tf.float32)
                self.compute_hidden = (tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias))
                self.compute_hidden_2 = tf.nn.sigmoid(tf.matmul(self.compute_hidden, self.w_2) + self.hidden_bias_2)
                z[i, j] = self.get_energy()
            print(i)
        return z

    def energy_dist(self, x):
        z = np.zeros(x.shape)
        for i in range(x.shape[0]):
            self.x = tf.constant([[0, 0, x[i]]], dtype=tf.float32)
            self.compute_hidden = (tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias))
            z[i] = self.get_energy()
        return z

    def transform_inv(self, batch_y):
        self.y = tf.Variable(batch_y, dtype=tf.float32)
        self.compute_visible = tf.matmul(self.y, tf.transpose(self.w)) + self.visible_bias
        return self.compute_visible.numpy()

    def reconstruct(self, batch_x):
        self.x = batch_x
        self._initialize_vars()
        err = self.get_err(batch_x)
        return self.compute_visible, err

    def get_weights(self):
        return self.w, self.visible_bias, self.hidden_bias

    def save_weights(self, prefix, overwrite=False):
        if overwrite:
            mode = 'w+'
        else:
            mode = 'ab'
        f = open("{}_weights.txt".format(prefix), mode)
        g = open("{}_visible_bias.txt".format(prefix), mode)
        h = open("{}_hidden_bias.txt".format(prefix), mode)
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

import tensorflow as tf
import numpy as np
from .dbm import DBM
from .util import sample_bernoulli, sample_gaussian, sample_laplace
import math

class GDBM(DBM):
    def __init__(self, n_visible, n_hidden, sample_visible=False, sigma=1, **kwargs):
        self.sample_visible = sample_visible
        self.stdevs = None
        self.means = None
        DBM.__init__(self, n_visible, n_hidden, **kwargs)

    def preprocess_data(self, confs):
        self.means = np.zeros(self.n_visible)
        self.stdevs = np.zeros(self.n_visible)
        for i in range(self.n_visible):
            self.means[i] = np.mean(confs[:,i])
            self.stdevs[i] = np.std(confs[:,i])
            confs[:,i] = (confs[:,i] - self.means[i]) / self.stdevs[i]
        return confs

    def postprocess_data(self, confs):
        return confs * self.stdevs + self.means

    def pretrain_first_layer(self):
        hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        visible_recon_p = tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.w)) + self.visible_bias

        if self.sample_visible:
            visible_recon_p = sample_gaussian(visible_recon_p, 0.7)

        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

        positive_grad = tf.matmul(tf.transpose(self.x), hidden_p)
        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        delta_w = positive_grad - negative_grad
        delta_visible_bias = tf.reduce_mean(self.x - visible_recon_p, 0)
        delta_hidden_bias = tf.reduce_mean(hidden_p - hidden_recon_p, 0)

        self.w = self.w + self.learning_rate * delta_w
        self.visible_bias = self.visible_bias + self.learning_rate * delta_visible_bias
        self.hidden_bias = self.hidden_bias + self.learning_rate * delta_hidden_bias

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.compute_visible = tf.matmul((self.compute_hidden),tf.transpose(self.w)) + self.visible_bias

    def pretrain_second_layer(self):
        hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias) # ???
        hidden_p_2 = tf.nn.sigmoid(tf.matmul(sample_bernoulli(hidden_p), self.w_2) + self.hidden_bias_2)
        hidden_recon_p = tf.nn.sigmoid(tf.matmul(sample_bernoulli(hidden_p_2), tf.transpose(self.w_2)) + self.hidden_bias)
        hidden_recon_p_2 = tf.nn.sigmoid(tf.matmul(sample_bernoulli(hidden_recon_p), self.w_2) + self.hidden_bias_2)

        positive_grad = tf.matmul(tf.transpose(hidden_p), hidden_p_2)
        negative_grad = tf.matmul(tf.transpose(hidden_recon_p), hidden_recon_p_2)

        delta_w_2 = positive_grad - negative_grad
        delta_hidden_bias = tf.reduce_mean(hidden_p - hidden_recon_p, 0)
        delta_hidden_bias_2 = tf.reduce_mean(hidden_p_2 - hidden_recon_p_2, 0)

        self.w_2 = self.w_2 + self.learning_rate * delta_w_2
        self.hidden_bias = self.hidden_bias + self.learning_rate * delta_hidden_bias
        self.hidden_bias_2 = self.hidden_bias_2 + self.learning_rate * delta_hidden_bias_2

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.compute_hidden_2 = tf.nn.sigmoid(tf.matmul((self.compute_hidden), self.w_2) + self.hidden_bias_2)
        self.compute_hidden = tf.nn.sigmoid(tf.matmul((self.compute_hidden_2), tf.transpose(self.w_2)) + self.hidden_bias)
        self.compute_visible = tf.matmul((self.compute_hidden), tf.transpose(self.w)) + self.visible_bias

    def global_optimization(self):
        ### Variational inference
        self.inference()

        ### Adjusting recognition parameters
        #delta_r1 = tf.matmul(tf.transpose(self.x),(hidden_inference - self.compute_hidden) * self.compute_hidden * (1 - self.compute_hidden))
        #delta_r2 = tf.matmul(tf.transpose(self.compute_hidden),(hidden_2_inference - self.compute_hidden_2) * self.compute_hidden_2 * (1 - self.compute_hidden_2))
        #self.r1 = self.r1 + self.learning_rate * delta_r1
        #self.r2 = self.r2 + self.learning_rate * delta_r2

        ### Stochastic approximation
        #visible_recon_p = tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias
        #hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias + tf.matmul(self.compute_hidden_2, tf.transpose(self.w_2)))
        #hidden_2_recon_p = tf.nn.sigmoid(tf.matmul(hidden_recon_p, self.w_2) + self.hidden_bias_2)

        #for i in range(self.mf_steps):
        #    visible_recon_p = tf.matmul(hidden_recon_p, tf.transpose(self.w)) + self.visible_bias
        #    hidden_recon_p = tf.nn.sigmoid(
        #        tf.matmul(visible_recon_p, self.w) + self.hidden_bias + tf.matmul(hidden_2_recon_p,
        #                                                                          tf.transpose(self.w_2)))
        #    hidden_2_recon_p = tf.nn.sigmoid(tf.matmul(hidden_recon_p, self.w_2) + self.hidden_bias_2)

        ### Update parameters
        #self.w = self.w + 0.01 * self.learning_rate * (tf.matmul(tf.transpose(self.x), self.compute_hidden) - tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p))
        #self.w_2 = self.w_2 + 0.01 * self.learning_rate * (tf.matmul(tf.transpose(self.compute_hidden), self.compute_hidden_2) - tf.matmul(tf.transpose(hidden_recon_p), hidden_2_recon_p))
        #self.learning_rate = self.learning_rate * 0.9
        #self.compute_visible = tf.nn.sigmoid(tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias)

    def inference(self):
        self.compute_hidden = tf.nn.sigmoid(2 * tf.matmul(self.x, self.w) + self.hidden_bias)
        self.compute_hidden_2 = tf.nn.sigmoid(tf.matmul(self.compute_hidden, self.w_2) + self.hidden_bias_2)

        for k in range(self.mf_steps):
            hidden_inference = tf.nn.sigmoid(tf.matmul(self.x, self.w) + tf.matmul(self.compute_hidden_2, tf.transpose(self.w_2)) + self.hidden_bias)
            hidden_2_inference = tf.nn.sigmoid(tf.matmul(self.compute_hidden, self.w_2) + self.hidden_bias_2)

        self.compute_hidden = hidden_inference
        self.compute_hidden_2 = hidden_2_inference


    def get_energy(self):
        self.sigma = 1.0
        return (tf.reduce_sum(tf.square(self.x) / (2 * np.square(self.sigma)))
                - tf.reduce_sum(self.hidden_bias * self.compute_hidden)
                - tf.matmul(tf.matmul(self.x / self.sigma, self.w), tf.transpose(self.compute_hidden))
                - tf.reduce_sum(self.hidden_bias_2 * self.compute_hidden_2)
                - tf.matmul(tf.matmul(self.compute_hidden, self.w_2), tf.transpose(self.compute_hidden_2)))

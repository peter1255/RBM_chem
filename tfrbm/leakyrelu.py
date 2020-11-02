import tensorflow as tf
import numpy as np
from .rbm import RBM
from .util import sample_bernoulli, sample_gaussian, sample_laplace
import math

class LeakyReluRBM(RBM):
    def __init__(self, n_visible, n_hidden, sample_visible=True, sigma=1, **kwargs):
        self.sample_visible = sample_visible
        self.stdevs = None
        self.sigma = sigma
        self.means = None
        self.chain = tf.zeros([1,n_visible])
        RBM.__init__(self, n_visible, n_hidden, **kwargs)

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

    def _initialize_vars(self):
        hidden_p = tf.nn.leaky_relu(tf.matmul(self.x, self.w) + self.hidden_bias)
        visible_recon_p = tf.matmul(hidden_p, tf.transpose(self.w)) + self.visible_bias

        if self.sample_visible:
            visible_recon_p = sample_gaussian(visible_recon_p, self.sigma)

        hidden_recon_p = tf.nn.leaky_relu(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

        positive_grad = tf.matmul(tf.transpose(self.x), hidden_p)
        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        delta_w = positive_grad - negative_grad
        delta_visible_bias = tf.reduce_mean(self.x - visible_recon_p, 0)
        delta_hidden_bias = tf.reduce_mean(hidden_p - hidden_recon_p, 0)

        self.w = self.w + self.learning_rate * delta_w
        self.visible_bias = self.visible_bias + self.learning_rate * delta_visible_bias
        self.hidden_bias = self.hidden_bias + self.learning_rate * delta_hidden_bias

        self._compute_hidden()
        self._compute_visible()

    def _compute_hidden(self):
        self.compute_hidden = tf.nn.leaky_relu(tf.matmul(self.x, self.w) + self.hidden_bias)

    def _compute_visible(self):
        self.compute_visible = tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias

    def get_energy(self, x=None, h=None):
        if x is None:
            x = self.x
        if h is None:
            h = self.compute_hidden
        self.sigma = 1.0
        return (tf.reduce_sum(tf.square(x - self.visible_bias) / (2 * np.square(self.sigma)))
                - tf.reduce_sum(self.hidden_bias * h)
                - tf.matmul(tf.matmul(x / self.sigma, self.w), tf.transpose(h)))
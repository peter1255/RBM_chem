import tensorflow as tf
from .rbm import RBM
from .util import sample_bernoulli


class BBRBM(RBM):
    def __init__(self, *args, **kwargs):
        RBM.__init__(self, *args, **kwargs)

    def _initialize_vars(self):
        def maximizer(slice_):
            return tf.where(
                tf.equal(tf.reduce_max(slice_, keepdims=True), slice_), tf.constant(1.0), tf.constant(0.0))

        def sample(samp):
            return tf.map_fn(
                maximizer, tf.convert_to_tensor(tf.split(samp, 3)))

        def fill_missing(x):  # keeps -1s
            return tf.where(tf.less(x[1], 0), -1, x[0])

        def fill_known(x):  # replaces -1s with predictions (keeps known data)
            return tf.where(tf.less(x[1], 0), x[0], x[1])

        def softmax(row):
            return tf.nn.softmax(tf.where(tf.greater_equal(row,0), row, -100000000000))

        if self.train:

            #convert visible layer to softmax
            soft_x = tf.map_fn(softmax, self.x)

            # sampling
            hidden_p = tf.nn.sigmoid(tf.matmul(soft_x, self.w) + self.hidden_bias)
            hidden = sample_bernoulli(hidden_p)
            visible_recon_p = tf.matmul(hidden, tf.transpose(self.w)) + self.visible_bias

            # parameters which are 0 (meaning no value has been given) stay at 0 during sampling
            elems = (visible_recon_p, self.x)
            visible_recon = tf.map_fn(fill_missing, elems, dtype=tf.float32)

            # softmax reconstructed visible layer
            softmax_visible_recon = tf.map_fn(softmax, visible_recon)

            # sample hidden layer again
            hidden_recon_p = tf.nn.sigmoid(tf.matmul(softmax_visible_recon, self.w) + self.hidden_bias)

            # contrastive divergence algorithm
            positive_grad = tf.matmul(tf.transpose(soft_x), hidden_p)
            negative_grad = tf.matmul(tf.transpose(softmax_visible_recon), hidden_recon_p)

            def f(x_old, x_new):
                return self.momentum * x_old + \
                       self.learning_rate * x_new * (1 - self.momentum) / tf.cast(tf.shape(x_new)[0], dtype=tf.float32)

            self.delta_w = f(self.delta_w, positive_grad - negative_grad)
            self.delta_visible_bias = f(self.delta_visible_bias, tf.reduce_mean(self.x - visible_recon_p, 0))
            self.delta_hidden_bias = f(self.delta_hidden_bias, tf.reduce_mean(hidden_p - hidden_recon_p, 0))

            self.w = self.w + self.delta_w
            self.visible_bias = self.visible_bias + self.delta_visible_bias
            self.hidden_bias = self.hidden_bias + self.delta_hidden_bias

            # compute visible nodes again for training accuracy
            hidden_p = tf.nn.sigmoid(tf.matmul(soft_x, self.w) + self.hidden_bias)
            hidden = sample_bernoulli(hidden_p)
            visible_recon_p = tf.matmul(hidden, tf.transpose(self.w)) + self.visible_bias

            self.compute_visible = tf.map_fn(sample, visible_recon_p)
            self.compute_visible = tf.reshape(self.compute_visible, [tf.shape(self.x)[0], 90])
            elems = (self.compute_visible, self.x)
            self.compute_visible = tf.map_fn(fill_missing, elems, dtype=tf.float32)

            '''
            hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
            visible_recon_p = tf.nn.sigmoid(tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.w)) + self.visible_bias)

            elems = (visible_recon_p, self.x)
            visible_recon_p = tf.map_fn(fill_missing, elems, dtype=tf.float32)

            hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

            positive_grad = tf.matmul(tf.transpose(self.x), hidden_p)
            negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

            def f(x_old, x_new):
                return self.momentum * x_old +\
                       self.learning_rate * x_new * (1 - self.momentum) / tf.cast(tf.shape(x_new)[0], dtype=tf.float32)

            delta_w_new = f(self.delta_w, positive_grad - negative_grad)
            delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(self.x - visible_recon_p, 0))
            delta_hidden_bias_new = f(self.delta_hidden_bias, tf.reduce_mean(hidden_p - hidden_recon_p, 0))

            update_delta_w = self.delta_w.assign(delta_w_new)
            update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
            update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

            update_w = self.w.assign(self.w + delta_w_new)
            update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
            update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

            self.update_deltas = [update_delta_w, update_delta_visible_bias, update_delta_hidden_bias]
            self.update_weights = [update_w, update_visible_bias, update_hidden_bias]

            self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
            self.compute_visible = tf.nn.sigmoid(tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias)

            self.compute_visible = tf.map_fn(sample, self.compute_visible)
            self.compute_visible = tf.reshape(self.compute_visible, [tf.shape(self.x)[0], 90])
            elems = (self.compute_visible, self.x)
            self.compute_visible = tf.map_fn(fill_missing, elems, dtype=tf.float32)
            trtest = self.compute_visible.numpy()
            '''
        else:  # test procedure

            #convert visible layer to softmax
            soft_x = tf.map_fn(softmax, self.x)

            # sampling
            hidden_p = tf.nn.sigmoid(tf.matmul(soft_x, self.w) + self.hidden_bias)
            hidden = sample_bernoulli(hidden_p)
            visible_recon_p = tf.matmul(hidden, tf.transpose(self.w)) + self.visible_bias

            # highest probability
            self.compute_visible = tf.map_fn(sample, visible_recon_p)
            self.compute_visible = tf.reshape(self.compute_visible, [tf.shape(self.x)[0], 90])

            # parameters which are 0 (meaning no value has been given) stay at 0 during sampling
            elems = (self.compute_visible, self.x)
            self.compute_visible = tf.map_fn(fill_known, elems, dtype=tf.float32)


            tttest = self.compute_visible.numpy()
            tttest = self.compute_visible.numpy()

        #self.compute_visible_from_hidden = tf.nn.sigmoid(tf.matmul(self.y, tf.transpose(self.w)) + self.visible_bias)


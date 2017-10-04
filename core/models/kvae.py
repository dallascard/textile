import tempfile

import sys
import numpy as np
import theano
import theano.tensor as T
from scipy.special import gamma

from ..models import theano_common

# set this to allow deep copy of model
sys.setrecursionlimit(5000)



class KAVE:

    def __init__(self, dv, nonlinearity='tanh', output_dir=None, name='model', pos_label=1):
        self._model_type = 'KVAE'
        self._nonlinearity = nonlinearity
        if output_dir is None:
            self._output_dir = tempfile.gettempdir()
        else:
            self._output_dir = output_dir
        self._name = name
        self._pos_label = pos_label
        self._train_f1 = None
        self._train_acc = None
        self._dev_f1 = None
        self._dev_acc = None
        self._dev_acc_cfm = None
        self._dev_pvc_cfm = None
        self._venn_info = None

        # create a variable to store the label proportions in the training data
        self._train_proportions = None
        # variable to hold the sklearn model
        self._model = None

class th_KVAE:

    def __init__(self, d_v, d_e, optimizer, optimizer_args, np_rng, th_rng, encoder_layers=1, clip_gradients=False, init_bias=None, train_bias=False, scale=6.0):

        self.d_v = d_v  # vocabulary size
        self.d_e = d_e  # dimensionality of encoder
        d_t = 1
        assert encoder_layers == 1 or encoder_layers == 2
        self.n_encoder_layers = encoder_layers

        # create parameter matrices and biases
        self.W_encoder_1 = theano_common.init_param('W_encoder_1', (d_e, d_v), np_rng, scale=scale)
        self.b_encoder_1 = theano_common.init_param('b_encoder_1', (d_e, ), np_rng, scale=0.0)

        self.W_encoder_2 = theano_common.init_param('W_encoder_2', (d_e, d_e), np_rng, scale=scale)
        self.b_encoder_2 = theano_common.init_param('b_encoder_2', (d_e, ), np_rng, scale=0.0)

        self.W_encoder_shortcut = theano_common.init_param('W_encoder_shortcut', (d_e, d_v), np_rng, scale=scale)

        self.W_a = theano_common.init_param('W_a', (1, d_e), np_rng, scale=scale)
        #self.b_mu = theano_common.init_param('b_a', (1, ), np_rng, scale=0.0)

        self.W_b = theano_common.init_param('W_b', (1, d_e), np_rng, scale=scale, values=np.zeros((1, d_e)))
        #self.b_sigma = theano_common.init_param('b_b', (1, ), np_rng, scale=0.0, values=np.array([-4] * 1))

        # create basic sets of parameters which we will use to tell the model what to update
        self.params = [self.W_encoder_1, self.b_encoder_1,
                       self.W_a,# self.b_mu,
                       self.W_b#, self.b_sigma
                       ]

        self.param_shapes = [(d_e, d_v), (d_e,),
                             (d_t, d_e), #(d_t,),
                             (d_t, d_e)#, (d_t,)
                             ]

        self.encoder_params = [self.W_encoder_1, self.b_encoder_1,
                               self.W_a, #self.b_mu,
                               self.W_b#, self.b_sigma
                               ]
        self.encoder_param_shapes = [(d_e, d_v), (d_e,),
                                     (d_t, d_e), #(d_t,),
                                     (d_t, d_e)#, (d_t,)
                                    ]

        # add encoder parameters depending on number of layers
        if self.n_encoder_layers > 1:
            self.params.extend([self.W_encoder_2, self.b_encoder_2])
            self.param_shapes.extend([(d_e, d_e), (d_e,)])
            self.encoder_params.extend([self.W_encoder_2, self.b_encoder_2])
            self.encoder_param_shapes.extend([(d_e, d_e), (d_e,)])

        # declare variables that will be given as inputs to functions to be declared below
        x = T.vector('x', dtype=theano.config.floatX)  # average of word vectors in document
        y = T.iscalar('y')  # labels for one item
        lr = T.fscalar('lr')  # learning rate
        l1_strength = T.fscalar('l1_strength')  # l1_strength
        kl_strength = T.fscalar('kl_strength')  # l1_strength

        # encode one item to mean and variance vectors
        a, b = self.encoder(x)

        # take a random sample from the corresponding multivariate normal
        p_sample = self.sampler(a, b, th_rng)

        # compute the KL divergence from the prior
        #KLD = -0.5 * T.sum(1 + log_sigma_sq - T.square(mu) - T.exp(log_sigma_sq))
        euler = 0.5772156649
        #KLD = T.mean((a - 1.0) / a * (-euler - T.psi(b) - 1.0/b) + T.log(a * b) - (b - 1.0) / b)
        KLD = T.mean((a - 1.0) / a * (-euler - self.psi(b) - 1.0/b) + T.log(a * b) - (b - 1.0) / b)

        # evaluate the likelihood
        #nll_term = -T.sum(T.log(p_sample[T.zeros(n_words, dtype='int32'), indices]) + 1e-32)
        nll_term = T.nnet.binary_crossentropy(p_sample, y).mean()

        # compute the loss
        #loss = nll_term + KLD
        loss = nll_term

        # compute gradients
        gradients = [T.cast(T.grad(loss, param, disconnected_inputs='warn'), dtype=theano.config.floatX) for param in self.params]
        #gradients = [T.cast(T.grad(nll_term, param, disconnected_inputs='warn'), dtype=theano.config.floatX) for param in self.params]

        # optionally clip gradients
        #if clip_gradients:
        #    gradients = theano_common.clip_gradients(gradients, 5)

        # create the updates for various sets of parameters
        updates = optimizer(self.params, self.param_shapes, gradients, lr, optimizer_args)
        #updates = []

        # declare the available methods for this class
        #self.test_input = theano.function(inputs=[x, indices], outputs=[n_words_print, x_sum])
        self.train = theano.function(inputs=[x, y, lr, l1_strength, kl_strength], outputs=[nll_term, KLD], updates=updates, on_unused_input='ignore')
        self.compute_loss = theano.function(inputs=[x, y, lr, l1_strength, kl_strength], outputs=[p_sample, nll_term, KLD], on_unused_input='ignore')
        #self.train_encoder = theano.function(inputs=[x, indices, y, lr, l1_strength, kl_strength], outputs=[nll_term, KLD, penalty], updates=encoder_updates, on_unused_input='ignore')
        #self.train_generator = theano.function(inputs=[x, indices, y, lr, l1_strength, kl_strength], outputs=[nll_term, KLD, penalty], updates=generator_updates, on_unused_input='ignore')
        #self.train_decoder = theano.function(inputs=[x, indices, y, lr, l1_strength, kl_strength], outputs=[nll_term, KLD, penalty], updates=decoder_updates, on_unused_input='ignore')
        #self.train_not_decoder = theano.function(inputs=[x, indices, y, lr, l1_strength, kl_strength], outputs=[nll_term, KLD, penalty], updates=other_updates, on_unused_input='ignore')
        #self.pretrain_decoder = theano.function(inputs=[indices, y, pretrain_r, lr, l1_strength, kl_strength], outputs=[pretrain_loss], updates=pretrain_updates, on_unused_input='ignore')
        self.encode = theano.function(inputs=[x], outputs=[a, b], on_unused_input='ignore')
        #self.decode = theano.function(inputs=[pretrain_r, y], outputs=[p_x_given_pretrain_h], on_unused_input='ignore')
        #self.sample = theano.function(inputs=[x, y], outputs=h, on_unused_input='ignore')
        #self.get_mean_doc_rep = theano.function(inputs=[x, y], outputs=r_mu, on_unused_input='ignore')
        #self.encode_and_decode = theano.function(inputs=[x, y], outputs=p_x_given_x, on_unused_input='ignore')
        #self.neg_log_likelihood = theano.function(inputs=[x, indices, y], outputs=[nll_term, KLD], on_unused_input='ignore')
        #self.neg_log_likelihood_mu = theano.function(inputs=[x, indices, y], outputs=[nll_term_mu, KLD], on_unused_input='ignore')
        #self.train_label_only = theano.function(inputs=[indices, y, lr, l1_strength], outputs=[nll_term_y_only, penalty], updates=label_only_updates)
        #self.neg_log_likelihood_label_only = theano.function(inputs=[indices, y], outputs=nll_term_y_only)

    def encoder(self, x):
        if self.n_encoder_layers == 1:
            temp = T.dot(self.W_encoder_1, x) + self.b_encoder_1
            pi = T.tanh(temp)
        else:
            temp = T.dot(self.W_encoder_1, x) + self.b_encoder_1
            temp2 = T.tanh(temp)
            pi = T.tanh(T.dot(self.W_encoder_2, temp2) + self.b_encoder_2)

        a = T.exp(T.dot(self.W_a, pi))
        b = T.exp(T.dot(self.W_b, pi))
        #mu = T.dot(self.W_mu, pi) + self.b_mu
        #log_sigma_sq = T.dot(self.W_sigma, pi) + self.b_sigma
        return a, b

    def sampler(self, a, b, theano_rng):
        u = theano_rng.uniform(low=0.0, high=1.0)
        p = (1 - u ** (1.0/b)) ** (1.0/a)
        return p

    def psi(self, x):
        x = x + 6
        p = 1.0 / (x * x)
        p = (((0.004166666666667 * p - 0.003968253986254 ) * p + 0.008333333333333) * p - 0.083333333333333) * p
        p = p + T.log(x) - 0.5 / x - 1 / (x - 1) - 1 / (x - 2) - 1 / (x - 3) - 1 / (x - 4) - 1 / (x - 5) - 1 / (x - 6)
        return p

    """
    def save_params(self, filename):
        np.savez_compressed(filename,
                            d_v=self.d_v,
                            d_e=self.d_e,
                            d_t=self.d_t,
                            n_classes=self.n_classes,

                            n_encoder_layers=self.n_encoder_layers,
                            n_generator_layers=self.n_generator_layers,
                            generator_transform=self.generator_transform,
                            use_interactions=self.use_interactions,
                            encode_labels=self.encode_labels,
                            encoder_shortcut=self.encoder_shortcut,
                            generator_shortcut=self.generator_shortcut,

                            W_encoder_1=self.W_encoder_1.get_value(),
                            W_encoder_2=self.W_encoder_2.get_value(),
                            W_encoder_label=self.W_encoder_label.get_value(),
                            W_encoder_shortcut=self.W_encoder_shortcut.get_value(),

                            W_mu=self.W_mu.get_value(),
                            W_sigma=self.W_sigma.get_value(),

                            W_generator_1=self.W_generator_1.get_value(),
                            W_generator_2=self.W_generator_2.get_value(),
                            W_generator_3=self.W_generator_3.get_value(),
                            W_generator_4=self.W_generator_4.get_value(),


                            W_decoder=self.W_decoder.get_value(),
                            W_decoder_label=self.W_decoder_label.get_value(),
                            W_decoder_inter=self.W_decoder_inter.get_value(),

                            b_encoder_1=self.b_encoder_1.get_value(),
                            b_encoder_2=self.b_encoder_2.get_value(),
                            b_mu=self.b_mu.get_value(),
                            b_sigma=self.b_sigma.get_value(),
                            b_generator_1=self.b_generator_1.get_value(),
                            b_generator_2=self.b_generator_2.get_value(),
                            b_generator_3=self.b_generator_3.get_value(),
                            b_generator_4=self.b_generator_4.get_value(),
                            b_decoder=self.b_decoder.get_value(),)
    """

    """
    def load_params(self, filename):
        # load parameters
        params = np.load(filename)
        # load the rest of the parameters
        self.d_e = params['d_e']
        self.d_t = params['d_t']
        self.n_classes = params['n_classes']

        self.n_encoder_layers = params['n_encoder_layers']
        self.n_generator_layers = params['n_generator_layers']
        self.generator_transform = params['generator_transform']
        self.use_interactions = params['use_interactions']
        self.encode_labels = params['encode_labels']
        self.encoder_shortcut = params['encoder_shortcut']
        self.generator_shortcut = params['generator_shortcut']

        self.W_encoder_1.set_value(params['W_encoder_1'])
        self.W_encoder_2.set_value(params['W_encoder_2'])
        self.W_encoder_label.set_value(params['W_encoder_label'])
        self.W_encoder_shortcut.set_value(params['W_encoder_shortcut'])

        self.W_mu.set_value(params['W_mu'])
        self.W_sigma.set_value(params['W_sigma'])

        self.W_generator_1.set_value(params['W_generator_1'])
        self.W_generator_2.set_value(params['W_generator_2'])
        self.W_generator_3.set_value(params['W_generator_3'])
        self.W_generator_4.set_value(params['W_generator_4'])

        self.W_decoder.set_value(params['W_decoder'])
        self.W_decoder_label.set_value(params['W_decoder_label'])
        self.W_decoder_inter.set_value(params['W_decoder_inter'])

        self.b_encoder_1.set_value(params['b_encoder_1'])
        self.b_encoder_2.set_value(params['b_encoder_1'])
        self.b_mu.set_value(params['b_mu'])
        self.b_sigma.set_value(params['b_sigma'])
        self.b_generator_1.set_value(params['b_generator_1'])
        self.b_generator_2.set_value(params['b_generator_2'])
        self.b_generator_3.set_value(params['b_generator_3'])
        self.b_generator_4.set_value(params['b_generator_4'])
        self.b_decoder.set_value(params['b_decoder'])
    """


def test():

    # get optimizer
    optimizer, opti_params = theano_common.get_optimizer('adagrad')

    # get random variables
    np_rng, th_rng = theano_common.get_rngs(np.random.randint(low=0, high=630000))


    # generate some random data
    n = 200
    p = 100

    kvae = th_KVAE(p, int(p/5), optimizer, opti_params, np_rng, th_rng, 1)


    X = np.random.randint(low=0, high=2, size=(n, p))
    coefs = np.random.randn(p)
    print(coefs)
    probs = 1.0/(1 + np.exp(-np.dot(X, coefs)))
    y = np.zeros(n)
    for i in range(n):
        y[i] = np.random.binomial(n=1, p=probs[i], size=1)

    X_test = np.random.randint(low=0, high=2, size=(n, p))
    #X_test = np.eye(p)
    test_probs = 1.0/(1 + np.exp(-np.dot(X_test, coefs)))
    y_test = np.zeros(n)
    for i in range(n):
        y_test[i] = np.random.binomial(n=1, p=test_probs[i], size=1)
    y_test_prop = np.mean(y_test)
    print(y_test_prop)


    print("Before training")
    test_mse = compute_mse(kvae, X_test, test_probs)
    #test_mse = eval(kvae, X_test, y_test)
    #print("train mse = %0.4f; test mse = %0.4f" % (train_mse, test_mse))
    #eval(kvae, coefs, X_test, test_probs)

    print("Training")
    for epoch in range(50):
        running_nll = 0.0
        running_kld = 0.0
        for i in range(n):
            nll, kld = kvae.train(np.array(X[i, :], dtype=np.float32), y[i], 0.005, 0.0, 1.0)
            running_nll += nll
            running_kld += kld

        test_mse = compute_mse(kvae, X_test, test_probs)
        print("mean nll = %0.4f; mean kld = %0.4f; test mse = %0.4f" % (running_nll / float(n), running_kld / float(n), test_mse))

    eval_eye(kvae, coefs)

        #train_mse = eval(kvae, X[:10, :], y[:10])
        #test_mse = eval(kvae, X_test, y_test)
        #print("train mse = %0.4f; test mse = %0.4f" % (train_mse, test_mse))


def kmedian(a, b):
    return (1.0 - 2.0 ** (-1.0/b)) ** (1.0/a)


def kmean(a, b):
    return b * gamma(1 + 1.0/a) * gamma(b) / gamma(1 + 1.0/a + b)


def compute_mse(kvae, X_test, y_test):
    mse = 0.0
    n_test, p = X_test.shape
    for i in range(n_test):
        a, b = kvae.encode(np.array(X_test[i, :], dtype=np.float32))
        pred = kmean(a, b)
        mse += (pred - y_test[i]) ** 2
    mse = mse / float(n_test)
    return mse


def eval(kvae, coefs, X_test, test_probs):
    mse = 0.0
    n_test, p = X_test.shape
    for i in range(n_test):
        a, b = kvae.encode(np.array(X_test[i, :], dtype=np.float32))
        pred = kmean(a, b)
        #mse += (pred - y_test[i]) ** 2
        print(coefs[i], test_probs[i], pred)

def eval_eye(kvae, coefs):
    p = len(coefs)
    X = np.eye(p)
    probs = 1.0/(1 + np.exp(-np.dot(X, coefs)))
    for i in range(p):
        a, b = kvae.encode(np.array(X[i, :], dtype=np.float32))
        pred = kmean(a, b)[0]
        print(coefs[i], probs[i], a, b, pred)


if __name__ == '__main__':
    test()
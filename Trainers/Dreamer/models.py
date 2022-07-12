import tensorflow.keras.layers as tfkl
from tensorflow_probability import distributions as tfd
import tensorflow as tf
from collections import namedtuple

# heavily relies on https://github.com/danijar/dreamer

State = namedtuple('State', 'mean std deter stoch')


class DenseDecoder:
    def __init__(self):
        pass


class RSSM:
    def __init__(self, stochastic, determenistic, hidden, checkpoint_dir):
        self.ckpt_dir = checkpoint_dir
        self.s_size = stochastic
        self.d_size = determenistic
        self.hidden = hidden
        self.cell = tfkl.GRUCell(self.d_size)
        self.prior_layers = [
            tfkl.Dense(hidden, tf.nn.elu, name='prior1'),
            tfkl.Dense(hidden, tf.nn.elu, name='prior2'),
            tfkl.Dense(2 * determenistic)
        ]
        self.posterior_layers = [
            tfkl.Dense(hidden, tf.nn.elu, name='post1'),
            tfkl.Dense(2 * determenistic)
        ]

    @tf.function
    def posterior(self, prev_state: State, prev_action, env_embedding):
        prior = self.prior(prev_state, prev_action)
        x = tf.concat(prior.deter, env_embedding, -1)
        x = self.posterior_layers[0](x)
        x = self.posterior_layers[1](x)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = tfd.MultivariateNormalDiag(mean, std).sample()
        post = State(
            mean=mean, std=std, deter=prior.deter, stoch=stoch
        )
        return post, prior

    @tf.function
    def prior(self, prev_state: State, prev_action) -> State:
        inp = tf.concat([prev_state.stoch, prev_action], -1)
        x = self.prior_layers[0](inp)
        x, det = self.cell(x, prev_state.deter)
        x = self.prior_layers[1](x)
        x = self.prior_layers[2](x)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = tfd.MultivariateNormalDiag(mean, std).sample()
        prior = State(
            mean=mean, std=std, deter=det, stoch=stoch
        )
        return prior

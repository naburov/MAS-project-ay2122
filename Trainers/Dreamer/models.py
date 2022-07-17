from typing import Tuple, List

import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import tensorflow as tf
from collections import namedtuple
import numpy as np

# heavily relies on https://github.com/danijar/dreamer

State = namedtuple('State', 'mean std deter stoch')

NUM_ACTIONS = 22


class SampleDist:

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return tf.reduce_mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return tf.gather(sample, tf.argmax(logprob))[0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -tf.reduce_mean(logprob, 0)


class ActionDecoder:
    def __init__(self, inp, size, units, act=tf.nn.elu,
                 min_std=1e-4, init_std=5, mean_scale=5):
        self.size = size
        self.units = units
        self.act = act
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale

        self.model = tf.keras.models.Sequential([
            tfkl.Input(shape=(inp,)),
            tfkl.Dense(units, tf.nn.elu),
            tfkl.Dense(units, tf.nn.elu),
            tfkl.Dense(2 * size)
        ])

    def backward(self, optimizer: tf.keras.optimizers.Optimizer, tape: tf.GradientTape, loss):
        grads = tape.gradient(loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def __call__(self, s: tf.Tensor) -> SampleDist:
        raw_init_std = np.log(np.exp(self.init_std) - 1)
        x = self.model(s)
        # https://www.desmos.com/calculator/rcmcf5jwe7
        mean, std = tf.split(x, 2, -1)
        mean = self.mean_scale * tf.tanh(mean / self.mean_scale)
        std = tf.nn.softplus(std + raw_init_std) + self.min_std
        dist = tfd.Normal(mean, std)
        dist = tfd.TransformedDistribution(dist, 'tanh')
        dist = tfd.Independent(dist, 1)
        dist = SampleDist(dist)
        return dist


class DenseEncoder(object):
    def __init__(self, obs_shape, emb_size, units):
        self.model = tf.keras.models.Sequential([
            tfkl.Input(shape=obs_shape),
            tfkl.Dense(units, tf.nn.elu),
            tfkl.Dense(units, tf.nn.elu),
            tfkl.Dense(emb_size)
        ])

    def backward(self, optimizer: tf.keras.optimizers.Optimizer, tape: tf.GradientTape, loss):
        grads = tape.gradient(loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def __call__(self, data: tf.Tensor) -> tf.Tensor:
        return self.model(data)


class DenseDecoder:
    def __init__(self, inp, units, out_size):
        self.model = tf.keras.models.Sequential([
            tfkl.Input(shape=(inp,)),
            tfkl.Dense(units, tf.nn.elu),
            tfkl.Dense(units, tf.nn.elu),
            tfkl.Dense(out_size)
        ])

    def __call__(self, features: tf.Tensor) -> tfp.distributions.Distribution:
        x = features
        x = self.model(x)
        return tfd.Independent(tfd.Normal(x, 1), len(()))

    def backward(self, optimizer: tf.keras.optimizers.Optimizer, tape: tf.GradientTape, loss):
        grads = tape.gradient(loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))


class RSSM:
    def __init__(self, stochastic, determenistic, hidden, checkpoint_dir):
        self.ckpt_dir = checkpoint_dir
        self.s_size = stochastic
        self.d_size = determenistic
        self.hidden = hidden
        self.prior_model = self.get_prior_model(hidden, determenistic, (None, NUM_ACTIONS), (None, determenistic,))
        self.post_model = self.get_post_model(hidden, determenistic, (None, NUM_ACTIONS), (None, determenistic,))

    def load(self, checkpoints_path):
        pass

    def save(self, checkpoints_path):
        pass

    def initial_state(self, batch_size) -> State:
        return State(
            mean=tf.zeros([batch_size, self.s_size]),
            std=tf.zeros([batch_size, self.s_size]),
            stoch=tf.zeros([batch_size, self.s_size]),
            deter=self.prior_model.get_layer('cell').get_initial_state(None, batch_size)
        )

    def backward(self, optimizer: tf.keras.optimizers.Optimizer, tape: tf.GradientTape, loss):
        grads = tape.gradient(loss, self.prior_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.prior_model.trainable_weights))

        grads = tape.gradient(loss, self.post_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.post_model.trainable_weights))

    def get_prior_model(self, hid, d_size, action_shape, deter_shape) -> tf.keras.Model:
        det_inputs = tf.keras.Input(shape=deter_shape)
        act_inputs = tf.keras.Input(shape=action_shape)
        x = tfkl.Concatenate.concat(axis=-1)([det_inputs, act_inputs])
        x, det_tensor = tfkl.GRUCell(d_size, name='cell')(x, det_inputs)
        x = tfkl.Dense(hid, tf.nn.elu, name='prior1')(x)
        x = tfkl.Dense(hid, tf.nn.elu, name='prior2')(x)
        x = tfkl.Dense(2 * d_size)(x)
        return tf.keras.Model(inputs=[det_inputs, act_inputs], outputs=[x, det_tensor])

    def get_post_model(self, hid, d_size, deter_shape, env_shape) -> tf.keras.Model:
        prior_inputs = tf.keras.Input(shape=deter_shape)
        env_inputs = tf.keras.Input(shape=env_shape)
        x = tfkl.Concatenate.concat(axis=-1)([prior_inputs, env_inputs])
        x = tfkl.Dense(hid, tf.nn.elu, name='post1')(x)
        x = tfkl.Dense(hid, tf.nn.elu, name='post2')(x)
        x = tfkl.Dense(2 * d_size)(x)
        return tf.keras.Model(inputs=[prior_inputs, env_inputs], outputs=[x])

    def observe(self, embedding, actions, state) -> Tuple[State, State]:
        res = static_scan(
            self.posterior, (state, actions, embedding), 10
        )
        prior = State(
            mean=tf.stack([p[0].mean for p in res]),
            std=tf.stack([p[0].std for p in res]),
            deter=tf.stack([p[0].deter for p in res]),
            stoch=tf.stack([p[0].stoch for p in res])
        )
        post = State(
            mean=tf.stack([p[1].mean for p in res]),
            std=tf.stack([p[1].std for p in res]),
            deter=tf.stack([p[1].deter for p in res]),
            stoch=tf.stack([p[1].stoch for p in res])
        )

        return prior, post

    @tf.function
    def posterior(self, prev_state: State, prev_action, env_embedding) -> Tuple[State, State]:
        prior = self.prior(prev_state, prev_action)
        x = self.post_model(prior.deter, env_embedding)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = tfd.MultivariateNormalDiag(mean, std).sample()
        post = State(
            mean=mean, std=std, deter=prior.deter, stoch=stoch
        )
        return prior, post

    @tf.function
    def prior(self, prev_state: State, prev_action) -> State:
        x, det = self.prior_model([prev_state.deter], prev_action)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = tfd.MultivariateNormalDiag(mean, std).sample()
        prior = State(
            mean=mean, std=std, deter=det, stoch=stoch
        )
        return prior


def static_scan(fn, start, n_iterations, *args, **kwargs):
    res = [start]
    for i in range(n_iterations):
        res.append(
            fn(*res[-1], *args, *kwargs)
        )
    return res

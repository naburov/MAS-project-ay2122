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

    def __call__(self, s):
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

    def __call__(self, data):
        return self.model(data)


class DenseDecoder:
    def __init__(self, inp, units, out_size):
        self.model = tf.keras.models.Sequential([
            tfkl.Input(shape=(inp,)),
            tfkl.Dense(units, tf.nn.elu),
            tfkl.Dense(units, tf.nn.elu),
            tfkl.Dense(out_size)
        ])

    def __call__(self, features):
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

    def backward(self, optimizer: tf.keras.optimizers.Optimizer, tape: tf.GradientTape, loss):
        grads = tape.gradient(loss, self.prior_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.prior_model.trainable_weights))

        grads = tape.gradient(loss, self.post_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.post_model.trainable_weights))

    def get_prior_model(self, hid, d_size, action_shape, deter_shape):
        det_inputs = tf.keras.Input(shape=deter_shape)
        act_inputs = tf.keras.Input(shape=action_shape)
        x = tfkl.Concatenate.concat(axis=-1)([det_inputs, act_inputs])
        x, det_tensor = tfkl.GRUCell(d_size)(x, det_inputs)
        x = tfkl.Dense(hid, tf.nn.elu, name='prior1')(x)
        x = tfkl.Dense(hid, tf.nn.elu, name='prior2')(x)
        x = tfkl.Dense(2 * d_size)(x)
        return tf.keras.Model(inputs=[det_inputs, act_inputs], outputs=[x, det_tensor])

    def get_post_model(self, hid, d_size, deter_shape, env_shape):
        prior_inputs = tf.keras.Input(shape=deter_shape)
        env_inputs = tf.keras.Input(shape=env_shape)
        x = tfkl.Concatenate.concat(axis=-1)([prior_inputs, env_inputs])
        x = tfkl.Dense(hid, tf.nn.elu, name='post1')(x)
        x = tfkl.Dense(hid, tf.nn.elu, name='post2')(x)
        x = tfkl.Dense(2 * d_size)(x)
        return tf.keras.Model(inputs=[prior_inputs, env_inputs], outputs=[x])

    def observe(self, embedding, actions):
        return None, None

    @tf.function
    def posterior(self, prev_state: State, prev_action, env_embedding):
        prior = self.prior(prev_state, prev_action)
        x = self.post_model(prior.deter, env_embedding)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = tfd.MultivariateNormalDiag(mean, std).sample()
        post = State(
            mean=mean, std=std, deter=prior.deter, stoch=stoch
        )
        return post, prior

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

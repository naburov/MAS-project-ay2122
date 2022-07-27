import os
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
VECTOR_OBS=97


class TanhBijector(tfp.bijectors.Bijector):

    def __init__(self, validate_args=False, name='tanh'):
        super().__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args,
            name=name)

    def _forward(self, x):
        return tf.nn.tanh(x)

    def _inverse(self, y):
        dtype = y.dtype
        y = tf.cast(y, tf.float32)
        y = tf.where(
            tf.less_equal(tf.abs(y), 1.),
            tf.clip_by_value(y, -0.99999997, 0.99999997), y)
        y = tf.atanh(y)
        y = tf.cast(y, dtype)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
        return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))


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

    def sample(self):
        return self._dist.sample()


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
        dist = tfd.TransformedDistribution(dist, TanhBijector())
        dist = tfd.Independent(dist, 1)
        dist = SampleDist(dist)
        return dist

    def save(self, checkpoint_dir, name):
        self.model.save(
            os.path.join(checkpoint_dir, name + '.h5')
        )

    def load(self, checkpoint_dir, name):
        self.model = tf.keras.models.load_model(
            os.path.join(checkpoint_dir, name + '.h5')
        )


class EnvDecoder:
    def get_model(self, embedding_size, env_memory_size, units, conv_filters):
        embedding_input = tf.keras.Input(shape=(embedding_size,))
        e = tf.keras.layers.Dense(units, activation=tf.nn.relu)(embedding_input)
        conv_branch = tf.keras.layers.Dense(units=units, activation=tf.nn.relu)(e)
        conv_branch = tf.keras.layers.Dense(units=units, activation=tf.nn.relu)(conv_branch)
        conv_branch = tf.keras.layers.Dense(units=11 * 11 * conv_filters, activation=tf.nn.relu)(conv_branch)
        conv_branch = tf.keras.layers.Reshape(target_shape=(11, 11, conv_filters))(conv_branch)
        conv_branch = tf.keras.layers.Conv2D(conv_filters, 1, padding='same', activation=tf.nn.relu)(conv_branch)
        conv_branch = tf.keras.layers.Conv2D(conv_filters, 1, padding='same', activation=tf.nn.relu)(conv_branch)
        tgt_field_output = tf.keras.layers.Conv2D(2 * env_memory_size, 1, padding='same', activation=None)(
            conv_branch)

        x = tf.keras.layers.Dense(units, activation=tf.nn.relu)(embedding_input)
        x = tf.keras.layers.Dense(units, activation=tf.nn.relu)(x)
        x = tf.keras.layers.Dense(units, activation=tf.nn.relu)(x)
        x = tf.keras.layers.Dense(units, activation=tf.nn.relu)(x)
        vector_output = tf.keras.layers.Dense(units=VECTOR_OBS * env_memory_size)(x)

        return tf.keras.Model(inputs=[embedding_input], outputs=[tgt_field_output, vector_output])

    def __init__(self, env_memory_size, emb_size, units, conv_filters):
        self.model = self.get_model(emb_size, env_memory_size, units, conv_filters)

    def backward(self, optimizer: tf.keras.optimizers.Optimizer, tape: tf.GradientTape, loss):
        grads = tape.gradient(loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def __call__(self, data: tf.Tensor) -> Tuple[tfd.Distribution, tfd.Distribution]:
        if len(data.shape) == 3:
            time_steps = data.shape[0]
            reshaped_data = tf.reshape(data, (-1, *data.shape[2:]))
            res = self.model(reshaped_data)
            res = tf.reshape(res[0], (time_steps, -1, *res[0].shape[1:])), tf.reshape(res[1], (time_steps, -1,
                                                                                               *res[1].shape[1:]))
        else:
            res = self.model(data)
        return tfd.Independent(tfd.Normal(res[0], 1), len(res[0].shape)), \
               tfd.Independent(tfd.Normal(res[1], 1), len(res[1].shape))

    def save(self, checkpoint_dir, name):
        self.model.save(
            os.path.join(checkpoint_dir, name + '.h5')
        )

    def load(self, checkpoint_dir, name):
        self.model = tf.keras.models.load_model(
            os.path.join(checkpoint_dir, name + '.h5')
        )


class EnvEncoder:
    def get_model(self, env_memory_size, units, conv_filters, out_size):
        tgt_inputs = tf.keras.Input(shape=(11, 11, 2 * env_memory_size))
        vector_input = tf.keras.Input(shape=(VECTOR_OBS * env_memory_size,))

        cv_kwargs = {
            'strides': 1, 'padding': 'same', 'activation': tf.nn.relu
        }
        conv_branch = tf.keras.layers.Conv2D(conv_filters, 3, **cv_kwargs)(tgt_inputs)
        conv_branch = tf.keras.layers.Conv2D(conv_filters * 2, 3, **cv_kwargs)(conv_branch)
        conv_branch = tf.keras.layers.Conv2D(conv_filters * 4, 3, **cv_kwargs)(conv_branch)
        conv_vector = tf.keras.layers.GlobalAveragePooling2D()(conv_branch)

        x = tf.keras.layers.Dense(units, activation=tf.nn.elu)(vector_input)
        x = tf.keras.layers.Dense(units, activation=tf.nn.elu)(x)
        x = tf.keras.layers.Dense(units, activation=tf.nn.elu)(x)
        x = tf.keras.layers.Dense(units, activation=tf.nn.elu)(x)
        concated = tf.keras.layers.Concatenate()([x, conv_vector])

        out = tf.keras.layers.Dense(out_size, activation=tf.nn.relu)(concated)
        return tf.keras.Model(inputs=[tgt_inputs, vector_input], outputs=out)

    def __init__(self, env_memory_size, emb_size, units, conv_filters):
        self.model = self.get_model(env_memory_size, units, conv_filters, emb_size)

    def backward(self, optimizer: tf.keras.optimizers.Optimizer, tape: tf.GradientTape, loss):
        grads = tape.gradient(loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def __call__(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        if len(data[1].shape) == 3:
            time_steps = data[0].shape[0]
            reshaped_data = tf.reshape(data[0], (-1, *data[0].shape[2:])), tf.reshape(data[1], (-1, *data[1].shape[2:]))
            res = self.model(reshaped_data)
            res = tf.reshape(res, (time_steps, -1, *res.shape[1:]))
        else:
            res = self.model(data)
        return res

    def save(self, checkpoint_dir, name):
        self.model.save(
            os.path.join(checkpoint_dir, name + '.h5')
        )

    def load(self, checkpoint_dir, name):
        self.model = tf.keras.models.load_model(
            os.path.join(checkpoint_dir, name + '.h5')
        )


class DenseDecoder:
    def __init__(self, inp, units, out_size):
        self.model = tf.keras.models.Sequential([
            tfkl.Input(shape=(inp,)),
            tfkl.Dense(units, tf.nn.elu),
            tfkl.Dense(units, tf.nn.elu),
            tfkl.Dense(units, tf.nn.elu),
            tfkl.Dense(out_size)
        ])

    def __call__(self, features: tf.Tensor) -> tfp.distributions.Distribution:
        if len(features.shape) == 3:
            time_steps = features.shape[0]
            reshaped_data = tf.reshape(features, (-1, *features.shape[2:]))
            res = self.model(reshaped_data)
            res = tf.reshape(res, (time_steps, -1, *res.shape[1:]))
        else:
            res = self.model(features)
        return tfd.Independent(tfd.Normal(res, 1), len(()))

    def backward(self, optimizer: tf.keras.optimizers.Optimizer, tape: tf.GradientTape, loss):
        grads = tape.gradient(loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def save(self, checkpoint_dir, name):
        self.model.save(
            os.path.join(checkpoint_dir, name + '.h5')
        )

    def load(self, checkpoint_dir, name):
        self.model = tf.keras.models.load_model(
            os.path.join(checkpoint_dir, name + '.h5')
        )


class RSSM:
    def __init__(self, stochastic, determenistic, embedding_shape, hidden):
        self.s_size = stochastic
        self.d_size = determenistic
        self.hidden = hidden
        self.prior_model = self.get_prior_model(hid=hidden, d_size=determenistic, action_shape=NUM_ACTIONS,
                                                stoch_shape=stochastic)
        self.post_model = self.get_post_model(hidden, determenistic, embedding_shape)

    def save(self, checkpoint_dir, prior_model_name, post_model_name):
        self.prior_model.save(
            os.path.join(checkpoint_dir, prior_model_name + '.h5')
        )
        self.post_model.save(
            os.path.join(checkpoint_dir, post_model_name + '.h5')
        )

    def load(self, checkpoint_dir, prior_model_name, post_model_name):
        self.prior_model = tf.keras.models.load_model(
            os.path.join(checkpoint_dir, prior_model_name + '.h5')
        )
        self.post_model = tf.keras.models.load_model(
            os.path.join(checkpoint_dir, post_model_name + '.h5')
        )

    def initial_state(self, batch_size) -> State:
        return State(
            mean=tf.zeros([batch_size, self.s_size], dtype=tf.float32),
            std=tf.zeros([batch_size, self.s_size], dtype=tf.float32),
            stoch=tf.zeros([batch_size, self.s_size], dtype=tf.float32),
            deter=self.prior_model.get_layer('cell').get_initial_state(None, batch_size, tf.float32)
        )

    def backward(self, optimizer: tf.keras.optimizers.Optimizer, tape: tf.GradientTape, loss):
        grads = tape.gradient(loss, self.prior_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.prior_model.trainable_weights))

        grads = tape.gradient(loss, self.post_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.post_model.trainable_weights))

    def get_prior_model(self, hid, d_size, action_shape, stoch_shape) -> tf.keras.Model:
        stoch_inputs = tf.keras.Input(shape=stoch_shape)
        det_inputs = tf.keras.Input(shape=d_size)
        act_inputs = tf.keras.Input(shape=action_shape)
        x = tfkl.Concatenate(axis=-1)([stoch_inputs, act_inputs])
        x = tfkl.Dense(hid, tf.nn.elu, name='prior0')(x)
        x, det_tensor = tfkl.GRUCell(d_size, name='cell')(x, det_inputs)
        x = tfkl.Dense(hid, tf.nn.elu, name='prior1')(x)
        x = tfkl.Dense(hid, tf.nn.elu, name='prior2')(x)
        x = tfkl.Dense(2 * self.s_size)(x)
        return tf.keras.Model(inputs=[det_inputs, stoch_inputs, act_inputs], outputs=[x, det_tensor])

    def get_post_model(self, hid, deter_shape, env_shape) -> tf.keras.Model:
        prior_inputs = tf.keras.Input(shape=deter_shape)
        env_inputs = tf.keras.Input(shape=env_shape)
        x = tfkl.Concatenate(axis=-1)([prior_inputs, env_inputs])
        x = tfkl.Dense(hid, tf.nn.elu, name='post1')(x)
        x = tfkl.Dense(hid, tf.nn.elu, name='post2')(x)
        x = tfkl.Dense(2 * self.s_size)(x)
        return tf.keras.Model(inputs=[prior_inputs, env_inputs], outputs=[x])

    def observe(self, embedding: tf.Tensor, actions: tf.Tensor, state: State) -> Tuple[State, State]:
        # shape (time_step, batch_size, dim)
        assert len(embedding.shape) == 3
        assert len(actions.shape) == 3

        if state is None:
            state = self.initial_state(embedding.shape[1])

        prior = []
        post = []
        for t in range(embedding.shape[0]):
            prior_state, posterior_state = self.posterior(prev_state=state,
                                                          prev_action=actions[t],
                                                          env_embedding=embedding[t])
            prior.append(prior_state)
            post.append(posterior_state)
            state = posterior_state

        prior = State(
            mean=tf.stack([p.mean for p in prior]),
            std=tf.stack([p.std for p in prior]),
            deter=tf.stack([p.deter for p in prior]),
            stoch=tf.stack([p.stoch for p in prior])
        )
        post = State(
            mean=tf.stack([p.mean for p in post]),
            std=tf.stack([p.std for p in post]),
            deter=tf.stack([p.deter for p in post]),
            stoch=tf.stack([p.stoch for p in post])
        )

        return prior, post

    @tf.function
    def posterior(self, prev_state: State, prev_action, env_embedding) -> Tuple[State, State]:
        prior = self.prior(prev_state, prev_action)
        x = self.post_model((prior.deter, env_embedding))
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = tfd.MultivariateNormalDiag(mean, std).sample()
        post = State(
            mean=mean, std=std, deter=prior.deter, stoch=stoch
        )
        return prior, post

    @tf.function
    def prior(self, prev_state: State, prev_action) -> State:
        x, det = self.prior_model((prev_state.deter, prev_state.stoch, prev_action))
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = tfd.MultivariateNormalDiag(mean, std).sample()
        prior = State(
            mean=mean, std=std, deter=det, stoch=stoch
        )
        return prior

import os
from typing import Callable

from Trainers.Dreamer.models import RSSM, DenseDecoder, ActionDecoder, EnvEncoder, EnvDecoder, State, static_scan
import tensorflow as tf
from tensorflow_probability import distributions as tfd

hidden = 100
determ = 30
stoch = 30
units = 100
num_actions = 22
kl_scale = 0.01
free_nats = 3.0
gamma = 0.99
lambda_ = 0.95
horizon = 15
env_memory_size = 8
embedding_size = 100
filters = 32


def get_feat(state: State) -> tf.Tensor:
    return tf.concat([state.stoch, state.deter], -1)


def get_dist(mean, std):
    return tfd.MultivariateNormalDiag(mean, std)


class Dreamer:
    def __init__(self, checkpoint_dir):
        self.encoder = EnvEncoder(env_memory_size, embedding_size, units, filters)
        print('___________________________')
        print(self.encoder.model.summary())
        self.dynamics_model = RSSM(stoch, determ, embedding_size, hidden)
        print('___________________________')
        print(self.dynamics_model.prior_model.summary())
        print(self.dynamics_model.post_model.summary())
        self.reward_model = DenseDecoder(stoch + determ, units, 1)
        print('___________________________')
        print(self.reward_model.model.summary())
        self.value_model = DenseDecoder(stoch + determ, units, 1)
        print('___________________________')
        print(self.value_model.model.summary())
        self.decoder_model = EnvDecoder(env_memory_size, stoch + determ, units, filters)
        print('___________________________')
        print(self.decoder_model.model.summary())
        self.action_model = ActionDecoder(stoch + determ, num_actions, units)
        print('___________________________')
        print(self.action_model.model.summary())

        self.action_opt = tf.keras.optimizers.Adam(2e-5)
        self.value_opt = tf.keras.optimizers.Adam(2e-5)
        self.dynamics_opt = tf.keras.optimizers.Adam(2e-5)
        self.reward_opt = tf.keras.optimizers.Adam(2e-5)
        self.encoder_opt = tf.keras.optimizers.Adam(2e-5)
        self.decoder_opt = tf.keras.optimizers.Adam(2e-5)

        if len(os.listdir(checkpoint_dir)) > 0:
            print('Loading checkpoints from {0}'.format(checkpoint_dir))
            self.load_state(checkpoint_dir)

    # @tf.function
    def policy(self, observations, state: State, prev_action: tf.Tensor, training=False):
        if state is None:
            latent = self.dynamics_model.initial_state(observations[0].shape[0])
            action = tf.zeros(shape=(observations[0].shape[0], num_actions))
        else:
            latent = state
            action = prev_action
        embedding = self.encoder(observations)
        _, post = self.dynamics_model.posterior(latent, action, embedding)
        features = get_feat(post)
        if not training:
            action = self.action_model(features).mode()
        else:
            action = self.action_model(features).sample()
            action = tf.clip_by_value(tfd.Normal(action, 0.1).sample(), 0, 1)
        return action, post

    def train_step(self, observations, actions, rewards):
        with tf.GradientTape(persistent=True) as model_tape:
            embed = self.encoder(observations)
            prior, post = self.dynamics_model.observe(embed, actions, state=None)
            feat = get_feat(post)
            vf_pred, v_pred = self.decoder_model(feat)
            reward_pred = self.reward_model(feat)
            reconstruction_loss = tf.math.log(
                tf.reduce_sum((observations[0] - vf_pred) ** 2) +
                tf.reduce_sum((observations[1] - v_pred) ** 2))
            reward_prob = tf.reduce_mean(reward_pred.log_prob(rewards[..., None]))

            prior_dist = get_dist(prior.stoch, prior.deter)
            post_dist = get_dist(post.stoch, post.deter)
            div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            div = tf.maximum(div, free_nats)
            model_loss = kl_scale * div + reconstruction_loss - reward_prob

        self.dynamics_model.backward(self.dynamics_opt, model_tape, model_loss)
        self.decoder_model.backward(self.decoder_opt, model_tape, model_loss)
        self.reward_model.backward(self.reward_opt, model_tape, model_loss)
        self.encoder.backward(self.encoder_opt, model_tape, model_loss)

        del model_tape

        with tf.GradientTape() as actor_tape:
            imag_feat = self.imagine_ahead(post)
            reward = self.reward_model(imag_feat).mode()
            value = self.value_model(imag_feat).mode()

            gamma_discount = tf.fill((value.shape[0] - 1, *value.shape[1:]), gamma)
            start_col = tf.ones((1, *value.shape[1:]))
            gamma_discount = tf.concat([start_col, gamma_discount], axis=0)
            gamma_discount = tf.math.cumprod(gamma_discount, 0)

            lambda_discount = tf.fill((value.shape[0] - 1, *value.shape[1:]), lambda_)
            start_col = tf.ones((1, *value.shape[1:]))
            lambda_discount = tf.concat([start_col, lambda_discount], axis=0)
            lambda_discount = tf.math.cumprod(lambda_discount, 0)

            reward_tensor = tf.multiply(gamma_discount, reward)
            value_tensor = tf.multiply(gamma_discount, value)
            reward_tensor = tf.cumsum(reward_tensor, 0)

            v_k_n = reward_tensor + value_tensor
            v_k_n = tf.multiply(lambda_discount, v_k_n)
            v_h_n = v_k_n[-1, ...]
            v_k_n = tf.cumsum(v_k_n[:-1], 0)[-1, ...]

            returns = (1 - lambda_) * v_k_n + v_h_n

            actor_loss = -tf.reduce_mean(returns)
        self.action_model.backward(self.action_opt, actor_tape, actor_loss)

        with tf.GradientTape() as value_tape:
            value_pred = self.value_model(imag_feat)[:-1]
            target = tf.stop_gradient(returns)
            value_loss = -tf.reduce_mean(value_pred.log_prob(target))
        self.value_model.backward(self.value_opt, value_tape, value_loss)

        return {
            'value_loss': value_loss.numpy(),
            'actor_loss': actor_loss.numpy(),
            'model_loss_kl_divergence': div.numpy(),
            'model_loss_reconstruction': reconstruction_loss.numpy(),
            'model_losss_reward_prob': reward_prob.numpy()
        }

    def imagine_ahead(self, post: State) -> tf.Tensor:

        state = State(
            mean=tf.reshape(post.mean, (-1, stoch)),
            std=tf.reshape(post.std, (-1, stoch)),
            stoch=tf.reshape(post.stoch, (-1, stoch)),
            deter=tf.reshape(post.mean, (-1, determ)),
        )

        prior_states = []
        for t in range(horizon):
            action = self.action_model(
                tf.stop_gradient(
                    get_feat(state)
                )).sample()
            prior = self.dynamics_model.prior(prev_state=state,
                                              prev_action=action)
            prior_states.append(prior)
            state = prior

        prior = State(
            mean=tf.squeeze(tf.stack([p.mean for p in prior_states])),
            std=tf.squeeze(tf.stack([p.std for p in prior_states])),
            deter=tf.squeeze(tf.stack([p.deter for p in prior_states])),
            stoch=tf.squeeze(tf.stack([p.stoch for p in prior_states]))
        )

        features = get_feat(prior)
        return features

    def save_state(self, checkpoint_dir):
        self.encoder.save(checkpoint_dir, 'encoder')
        self.dynamics_model.save(checkpoint_dir, 'prior', 'post')
        self.reward_model.save(checkpoint_dir, 'reward')
        self.value_model.save(checkpoint_dir, 'value')
        self.decoder_model.save(checkpoint_dir, 'decoder')
        self.action_model.save(checkpoint_dir, 'action')

    def load_state(self, checkpoint_dir):
        self.encoder.load(checkpoint_dir, 'encoder')
        self.dynamics_model.load(checkpoint_dir, 'prior', 'post')
        self.reward_model.load(checkpoint_dir, 'reward')
        self.value_model.load(checkpoint_dir, 'value')
        self.decoder_model.load(checkpoint_dir, 'decoder')
        self.action_model.load(checkpoint_dir, 'action')

    def v_k_n(self, reward, value) -> tf.Tensor:
        discount = gamma * tf.ones_like(reward)
        discount = tf.math.cumprod(discount, axis=0)
        reward = tf.math.cumsum(reward, axis=0)
        discounted_rewards = tf.multiply(discount, reward)
        discounted_values = tf.multiply(discount * gamma, value)
        return discounted_values + discounted_rewards

    def estimate_return(self, v_k_n: tf.Tensor):
        discount = lambda_ * tf.ones_like(v_k_n)
        discount = tf.math.cumprod(discount, axis=0)
        v_k_n = tf.math.cumsum(v_k_n, axis=0)
        discounted_v_k_n = tf.multiply(discount, v_k_n)
        return (1 - lambda_) * discounted_v_k_n

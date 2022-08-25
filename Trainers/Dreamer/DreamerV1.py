import os

from Trainers.Dreamer.models import RSSM, DenseDecoder, ActionDecoder, EnvEncoder, EnvDecoder, State
import tensorflow as tf
from config import *
from tensorflow_probability import distributions as tfd


def get_feat(state: State) -> tf.Tensor:
    return tf.concat([state.stoch, state.deter], -1)


def get_dist(mean, std):
    return tfd.MultivariateNormalDiag(mean, std)


class Dreamer:
    def __init__(self, checkpoint_dir):
        self.encoder = EnvEncoder(env_memory_size, embedding_size, units, filters)
        self.dynamics_model = RSSM(stoch, determ, embedding_size, hidden)
        self.reward_model = DenseDecoder(stoch + determ, units, 1, REWARD_DENSE_DECODER_NOISE)
        self.value_model = DenseDecoder(stoch + determ, units, 1, VALUE_DENSE_DECODER_NOISE)
        self.decoder_model = EnvDecoder(env_memory_size, stoch + determ, units, filters, DECODER_VF_STD, DECODER_V_STD)
        self.action_model_1 = ActionDecoder(stoch + determ, num_actions // 2, units)
        self.action_model_2 = ActionDecoder(stoch + determ, num_actions // 2, units)

        self.action_opt_1 = tf.keras.optimizers.Adam(ACTOR_LR)
        self.action_opt_2 = tf.keras.optimizers.Adam(ACTOR_LR)
        self.value_opt = tf.keras.optimizers.Adam(VALUE_LR)
        self.dynamics_opt = tf.keras.optimizers.Adam(DYNAMICS_LR)
        self.reward_opt = tf.keras.optimizers.Adam(REWARD_LR)
        self.encoder_opt = tf.keras.optimizers.Adam(ENCODER_LR)
        self.decoder_opt = tf.keras.optimizers.Adam(DECODER_LR)

        self.decoder_model.model.summary()

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
        prior, post = self.dynamics_model.posterior(latent, action, embedding)
        features = get_feat(post)
        if not training:
            action_1 = self.action_model_1(features).mode()
            action_2 = self.action_model_2(features).mode()
            action = tf.concat([action_1, action_2], axis=-1)
        else:
            action_1 = self.action_model_1(features).sample()
            action_2 = self.action_model_2(features).sample()
            action = tf.concat([action_1, action_2], axis=-1)
            action = tf.clip_by_value(
                (tfd.Normal(action, noise).sample() + 1.0) / 2,
                0, 1)
        return action, post

    def train_step(self, observations, actions, rewards):
        with tf.GradientTape(persistent=True) as model_tape:
            # print(tf.reduce_mean(rewards))
            embed = self.encoder(observations)
            prior, post = self.dynamics_model.observe(embed, actions, state=None)
            feat = get_feat(post)
            vf_pred, v_pred = self.decoder_model(feat)
            reward_pred = self.reward_model(feat)
            reconstruction_loss = tf.reduce_mean(vf_pred.log_prob(observations[0])) + tf.reduce_mean(
                v_pred.log_prob(observations[1]))
            reward_prob = tf.reduce_mean(reward_pred.log_prob(rewards))

            prior_dist = get_dist(prior.mean, prior.std)
            post_dist = get_dist(post.mean, post.std)
            div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            model_loss = kl_scale * div - rec_scale * reconstruction_loss - rew_scale * reward_prob

        self.dynamics_model.backward(self.dynamics_opt, model_tape, model_loss)
        self.decoder_model.backward(self.decoder_opt, model_tape, model_loss)
        self.reward_model.backward(self.reward_opt, model_tape, model_loss)
        self.encoder.backward(self.encoder_opt, model_tape, model_loss)

        del model_tape

        with tf.GradientTape(persistent=True) as actor_tape:
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
        self.action_model_1.backward(self.action_opt_1, actor_tape, actor_loss)
        self.action_model_2.backward(self.action_opt_2, actor_tape, actor_loss)

        del actor_tape

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
        def train_policy(s):
            f = tf.stop_gradient(
                get_feat(s)
            )
            action_1 = self.action_model_1(f).sample()
            action_2 = self.action_model_2(f).sample()
            action = tf.concat([action_1, action_2], axis=-1)
            a = (action + 1.0) / 2
            return a

        state = State(
            mean=tf.reshape(post.mean, (-1, stoch)),
            std=tf.reshape(post.std, (-1, stoch)),
            stoch=tf.reshape(post.stoch, (-1, stoch)),
            deter=tf.reshape(post.deter, (-1, determ)),
        )

        prior_states = []
        for t in range(horizon):
            action = train_policy(state)
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
        self.action_model_1.save(checkpoint_dir, 'action_1')
        self.action_model_2.save(checkpoint_dir, 'action_2')

    def load_state(self, checkpoint_dir):
        if os.path.exists(os.path.join(checkpoint_dir, 'encoder.h5')):
            self.encoder.load(checkpoint_dir, 'encoder')
        if os.path.exists(os.path.join(checkpoint_dir, 'prior.h5')):
            self.dynamics_model.load(checkpoint_dir, 'prior', 'post')
        if os.path.exists(os.path.join(checkpoint_dir, 'reward.h5')):
            self.reward_model.load(checkpoint_dir, 'reward')
        if os.path.exists(os.path.join(checkpoint_dir, 'value.h5')):
            self.value_model.load(checkpoint_dir, 'value')
        if os.path.exists(os.path.join(checkpoint_dir, 'decoder.h5')):
            self.decoder_model.load(checkpoint_dir, 'decoder')
        if os.path.exists(os.path.join(checkpoint_dir, 'action_1.h5')):
            self.action_model_1.load(checkpoint_dir, 'action_1')
        if os.path.exists(os.path.join(checkpoint_dir, 'action_2.h5')):
            self.action_model_2.load(checkpoint_dir, 'action_2')

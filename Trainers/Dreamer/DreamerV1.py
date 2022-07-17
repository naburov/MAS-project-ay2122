from Trainers.Dreamer.models import RSSM, DenseDecoder, ActionDecoder, DenseEncoder, State
import tensorflow as tf
from tensorflow_probability import distributions as tfd

hidden = 200
determ = 200
stoch = 200
units = 100
num_actions = 22
obs_shape = (585,)
rssm_checkpoint_dir = ''
kl_scale = 0.01
free_nats = 3.0
gamma = 0.99
lambda_ = 0.95


def get_feat(stoch, deter):
    return tf.concat([stoch, deter], -1)


def get_dist(mean, std):
    return tfd.MultivariateNormalDiag(mean, std)


class Dreamer:
    def __init__(self):
        self.encoder = DenseEncoder(obs_shape, 200, units)
        self.dynamics_model = RSSM(stoch, determ, hidden, rssm_checkpoint_dir)
        self.reward_model = DenseDecoder(400, units, 1)
        self.value_model = DenseDecoder(400, units, 1)
        self.decoder_model = DenseDecoder(400, units, 1)
        self.action_model = ActionDecoder(400, 200, units)

        self.action_opt = tf.keras.optimizers.Adam()
        self.value_opt = tf.keras.optimizers.Adam()
        self.dynamics_opt = tf.keras.optimizers.Adam()
        self.reward_opt = tf.keras.optimizers.Adam()
        self.encoder_opt = tf.keras.optimizers.Adam()
        self.decoder_opt = tf.keras.optimizers.Adam()

    @tf.function
    def policy(self, observations: tf.Tensor, state: State, prev_action: tf.Tensor):
        if state is None:
            latent = self.dynamics_model.initial_state(observations.shape[0])
            action = tf.zeros(shape=(observations.shape[0], num_actions))
        else:
            latent = state
            action = prev_action
        embedding = self.encoder(observations)
        _, post = self.dynamics_model.posterior(latent, action, embedding)
        features = get_feat(post.stoch, post.deter)
        action = self.action_model(features).mode()
        action = tf.clip_by_value(tfd.Normal(action, 0.1).sample(), -1, 1)
        return action, post

    def train_step(self, observations, actions, rewards):
        with tf.GradientTape() as model_tape:
            embed = self.encoder(observations)
            prior, post = self.dynamics_model.observe(embed, actions, state=None)
            feat = get_feat(post.stoch, post.deter)
            image_pred = self.decoder_model(feat)
            reward_pred = self.reward_model(feat)
            image_prob = tf.reduce_mean(image_pred.log_prob(observations))
            reward_prob = tf.reduce_mean(reward_pred.log_prob(rewards))

            prior_dist = get_dist(prior.stoch, prior.deter)
            post_dist = get_dist(post.stoch, post.deter)
            div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            div = tf.maximum(div, free_nats)
            model_loss = kl_scale * div - sum(image_prob + reward_prob)

        self.dynamics_model.backward(self.dynamics_opt, model_tape, model_loss)
        self.decoder_model.backward(self.decoder_opt, model_tape, model_loss)
        self.reward_model.backward(self.reward_opt, model_tape, model_loss)
        self.encoder.backward(self.encoder_opt, model_tape, model_loss)

        with tf.GradientTape() as actor_tape:
            imag_feat = self.imagine_ahead(post)
            reward = self.reward_model(imag_feat).mode()

            pcont = gamma * tf.ones_like(reward)
            value = self.value_model(imag_feat).mode()
            returns = self.estimate_return(reward, value)
            discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
                [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
            actor_loss = -tf.reduce_mean(discount * returns)
        self.action_model.backward(self.action_opt, actor_tape, actor_loss)

        with tf.GradientTape() as value_tape:
            value_pred = self.value_model(imag_feat)[:-1]
            target = tf.stop_gradient(returns)
            value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))
        self.value_model.backward(self.value_opt, value_tape, value_loss)

    def imagine_ahead(self, post) -> tf.Tensor:
        return None

    def save_state(self):
        pass

    def estimate_single_return(self, reward, value):
        discount = tf.stack([gamma * tf.ones_like(reward) for i in range(reward.shape[-1])])
        discount = tf.math.cumprod(discount, axis=-1)
        discounted_rewards = tf.multiply(discount, reward)
        discounted_values = tf.multiply(discount * gamma, value)
        return discounted_values + discounted_rewards

    def estimate_return(self, reward, value):
        return None

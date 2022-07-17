from Trainers.Dreamer.models import RSSM, DenseDecoder, ActionDecoder, DenseEncoder
import tensorflow as tf
from tensorflow_probability import distributions as tfd

hidden = 200
determ = 200
stoch = 200
units = 100
obs_shape = (585,)
rssm_checkpoint_dir = ''
kl_scale = 0.01
free_nats = 3.0
gamma = 0.99
lambd = 0.95


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

    def train_step(self, observations, actions, rewards):
        with tf.GradientTape() as model_tape:
            embed = self.encoder(observations)
            post, prior = self.dynamics_model.observe(embed, actions)
            feat = get_feat(post.stoch, post.deter)
            image_pred = self.decoder_model(feat)
            reward_pred = self.reward_model(feat)
            image_prob = tf.reduce_mean(image_pred.log_prob(observations))
            reward_prob = tf.reduce_mean(reward_pred.log_prob(rewards))
            if self._c.pcont:
                pcont_pred = self._pcont(feat)
                pcont_target = self._c.discount * data['discount']
                likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
                likes.pcont *= self._c.pcont_scale
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
            imag_feat = self._imagine_ahead(post)
            reward = self._reward(imag_feat).mode()
            if self._c.pcont:
                pcont = self._pcont(imag_feat).mean()
            else:
                pcont = self._c.discount * tf.ones_like(reward)
            value = self.value_model(imag_feat).mode()
            returns = tools.lambda_return(
                reward[:-1], value[:-1], pcont[:-1],
                bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
            discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
                [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
            actor_loss = -tf.reduce_mean(discount * returns)
        self.action_model.backward(self.action_opt, actor_tape, actor_loss)

        with tf.GradientTape() as value_tape:
            value_pred = self.value_model(imag_feat)[:-1]
            target = tf.stop_gradient(returns)
            value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))
        self.value_model.backward(self.value_opt, model_tape, value_loss)

    def save_state(self):
        pass

import tensorflow as tf
import numpy as np
from replay_buffer import ReplayBuffer

critic_lr = 0.0002
actor_lr = 0.0001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.9
# Used to update target networks
tau = 0.005


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


# @tf.function
def update(
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        target_actor,
        target_critic,
        critic_model,
        actor_model
):
    # Training and updating Actor & Critic networks.
    # See Pseudo Code.
    with tf.GradientTape() as tape:
        target_actions = target_actor(next_state_batch, training=True)
        y = reward_batch + gamma * target_critic(
            [next_state_batch, target_actions], training=True
        )
        critic_value = critic_model([state_batch, action_batch], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

    critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_grad, critic_model.trainable_variables)
    )

    with tf.GradientTape() as tape:
        actions = actor_model(state_batch, training=True)
        critic_value = critic_model([state_batch, actions], training=True)
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        actor_loss = -tf.math.reduce_mean(critic_value)

    actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(actor_grad, actor_model.trainable_variables)
    )


def policy(state, noise_object, actor_model):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, 0.0, 1.0)

    return [np.squeeze(legal_action)]


def train_step(
        batch_size,
        replay_buffer: ReplayBuffer,
        target_actor,
        target_critic,
        critic_model,
        actor_model
):
    vf, v, r, next_vf, next_v, a = replay_buffer.sample(batch_size)
    update((vf, v), a, r, (next_vf, next_v), target_actor, target_critic, critic_model, actor_model)

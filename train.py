from mpi4py import MPI
import numpy as np
import os

import statistics

from MyEnv import MyEnv
from noise_utils import OUActionNoise
from config import *
from logger import Logger

comm = MPI.COMM_WORLD
total_ranks = comm.Get_size()
rank = comm.Get_rank()
env = None
data = None
memory_size = 8
tau = 0.01
std_dev = 0.2
batch_size = 1024
buf_capacity = int(1e5)
checkpoint_dir = r'C:\Users\burov\Projects\mas-project-burov-ay2122\checkpoints'


#  TODO
#  1. save buf and model
#  2. get action method

def predict_actions(obss, actor, noise_object: OUActionNoise):
    vfs = np.stack([o[0] for o in obss], axis=0)
    vs = np.stack([o[1] for o in obss], axis=0)
    noise = noise_object.sample_batch(len(obss))
    pred_actions = actor((vfs, vs)).numpy() + noise
    legal_action = np.clip(pred_actions, 0., 1.)
    return legal_action


if rank == 0:
    import tensorflow as tf
    from models import get_critic_model, get_actor_model
    from replay_buffer import ReplayBuffer
    from train_utils import update_target, train_step

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    buf = ReplayBuffer(buf_capacity, memory_size)
    ou_noise = OUActionNoise(mean=np.full((22,), 0.5), std_deviation=0.5 * np.ones(1))
    logger = Logger()

    actor_model = get_actor_model(memory_size)
    critic_model = get_critic_model(memory_size)

    target_actor = get_actor_model(memory_size)
    target_critic = get_critic_model(memory_size)

    if os.path.exists(os.path.join(checkpoint_dir, 'actor.h5')):
        actor_model.load_weights(os.path.join(checkpoint_dir, 'actor.h5'))
        critic_model.load_weights(os.path.join(checkpoint_dir, 'critic_model.h5'))

        target_actor.load_weights(os.path.join(checkpoint_dir, 'target_actor.h5'))
        target_critic.load_weights(os.path.join(checkpoint_dir, 'target_critic.h5'))

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())
    if os.path.exists(os.path.join(checkpoint_dir, 'actor.h5')):
        buf.load('buf.pckl')
    for epoch in range(EPOCHS):
        for num_episode in range(EPISODES_PER_EPOCH):
            # init env
            cont_env = [i for i in range(1, total_ranks)]

            observations = [None] * len(cont_env)
            rewards = [0.0] * len(cont_env)

            print(cont_env)
            print(observations)
            print(rewards)

            for i in range(0, len(cont_env)):
                comm.send(0, dest=cont_env[i], tag=11)

            for i in range(0, len(cont_env)):
                data = comm.recv(source=cont_env[i], tag=11)
                observations[cont_env[i] - 1] = data['obs']

            while len(cont_env) > 0:
                # make step
                actions = predict_actions([observations[id - 1] for id in cont_env], actor_model, ou_noise)
                old_observations = observations
                observations = [None] * (total_ranks - 1)

                for i in range(0, len(cont_env)):
                    comm.send(1, dest=cont_env[i], tag=11)
                    comm.Send([actions[i], MPI.FLOAT], dest=cont_env[i], tag=77)

                ranks2del = []

                for i in range(0, len(cont_env)):
                    data = comm.recv(source=cont_env[i], tag=11)
                    observations[cont_env[i] - 1] = data['obs']
                    rewards[cont_env[i] - 1] += data['r']
                    buf.append((*old_observations[cont_env[i] - 1], data['r'], *data['obs'], actions[i]))

                    if data['done']:
                        ranks2del.append(cont_env[i])

                for r in ranks2del:
                    cont_env.remove(r)
            train_step(batch_size,
                       buf,
                       target_actor,
                       target_critic,
                       critic_model,
                       actor_model
                       )
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)
            print('Finished episode ', num_episode)
            avg = statistics.mean(rewards)
            logger.log2txt('reward_logs.txt',
                           'Epoch: {0}, Ep n: {1}, Avg rew: {2}'.format(epoch, num_episode, avg))

        actor_model.save_weights(os.path.join(checkpoint_dir, 'actor.h5'))
        critic_model.save_weights(os.path.join(checkpoint_dir, 'critic_model.h5'))

        target_actor.save_weights(os.path.join(checkpoint_dir, 'target_actor.h5'))
        target_critic.save_weights(os.path.join(checkpoint_dir, 'target_critic.h5'))
        if epoch % 10 == 0:
            buf.save('buf.pckl')

    for i in range(1, total_ranks):
        comm.send(2, dest=i, tag=11)
else:
    while True:
        data = comm.recv(source=0, tag=11)
        if data == 0:
            env = MyEnv(visualize=False, memory_size=memory_size)
            observation = env.reset()
            comm.send({
                'obs': observation,
                'r': 0,
                'done': False,
                'info': {}
            }, dest=0, tag=11)
        elif data == 1:
            action = np.empty(22, dtype=np.float)
            comm.Recv([action, MPI.FLOAT], source=0, tag=77)
            observation, reward, done, info = env.step(action)
            comm.send({
                'obs': observation,
                'r': reward,
                'done': done,
                'info': {}
            }, dest=0, tag=11)
        elif data == 2:
            break

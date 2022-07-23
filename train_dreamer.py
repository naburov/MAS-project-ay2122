from mpi4py import MPI
import numpy as np
import statistics

from MyEnv import MyEnv
from Trainers.Dreamer.TrainManager import DreamerTrainManager
from config import *
from Trainers.DDPGTrainManager.logger import Logger

comm = MPI.COMM_WORLD
total_ranks = comm.Get_size()
rank = comm.Get_rank()

env = None
data = None

if rank == 0:
    import tensorflow as tf

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

    manager = DreamerTrainManager(checkpoint_dir=MODEL_CHECKPOINT_DIR,
                                  buffer_path=BUFFER_CHECKPOINT_DIR,
                                  memory_size=ENV_MEMORY_SIZE,
                                  buffer_capacity=BUFFER_CAPACITY,
                                  num_ranks=total_ranks,
                                  batch_size=BATCH_SIZE)
    logger = Logger()
    for epoch in range(EPOCHS):
        for num_episode in range(EPISODES_PER_EPOCH):
            manager.on_episode_begin(epoch, num_episode)
            # init env
            cont_env = [i for i in range(1, total_ranks)]

            observations = [None] * len(cont_env)
            rewards = [0.0] * len(cont_env)

            for i in range(0, len(cont_env)):
                comm.send(0, dest=cont_env[i], tag=11)

            for i in range(0, len(cont_env)):
                data = comm.recv(source=cont_env[i], tag=11)
                observations[cont_env[i] - 1] = data['obs']

            while len(cont_env) > 0:
                # make step
                actions = manager.predict_actions([observations[id - 1] for id in cont_env])

                old_observations = observations
                observations = [None] * (total_ranks - 1)

                for i in range(0, len(cont_env)):
                    comm.send(1, dest=cont_env[i], tag=11)
                    comm.Send([actions[i], MPI.FLOAT], dest=cont_env[i], tag=77)

                ranks2del = []
                br = False

                for i in range(0, len(cont_env)):
                    data = comm.recv(source=cont_env[i], tag=11)
                    observations[cont_env[i] - 1] = data['obs']
                    rewards[cont_env[i] - 1] += data['r']
                    manager.append_observations(
                        (*old_observations[cont_env[i] - 1], data['r'], *data['obs'], actions[i]), data['info'])

                    if data['done']:
                        br = True

                if br:
                    break

            print('Finished episode ', num_episode)
            avg = statistics.mean(rewards)
            logger.log2txt(REWARD_LOGS_PATH,
                           'Epoch: {0}, Ep n: {1}, Avg rew: {2}'.format(epoch, num_episode, avg))

            manager.train_step()
        manager.on_epoch_end(epoch)

    for i in range(1, total_ranks):
        comm.send(2, dest=i, tag=11)
else:
    while True:
        data = comm.recv(source=0, tag=11)
        if data == 0:
            env = MyEnv(visualize=False, memory_size=ENV_MEMORY_SIZE)
            observation = env.reset()
            comm.send({
                'obs': observation,
                'r': 0,
                'done': False,
                'info': {'rank': rank}
            }, dest=0, tag=11)
        elif data == 1:
            action = np.empty(22, dtype=np.float)
            comm.Recv([action, MPI.FLOAT], source=0, tag=77)
            observation, reward, done, info = env.step(action)
            comm.send({
                'obs': observation,
                'r': reward,
                'done': done,
                'info': {'rank': rank}
            }, dest=0, tag=11)
        elif data == 2:
            break

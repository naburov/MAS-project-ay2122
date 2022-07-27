from mpi4py import MPI
import numpy as np
import statistics

from MyEnv import MyEnv
from Trainers.Dreamer.TrainManager import DreamerTrainManager
from config import *
from Trainers.DDPGTrainManager.logger import Logger
from tqdm import tqdm

comm = MPI.COMM_WORLD
total_ranks = comm.Get_size()
rank = comm.Get_rank()

env = None
data = None

train_steps = 25

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
    print('Loaded: ', manager.buf.num_sequences, ' sequences')
    for epoch in range(EPOCHS):
        is_random = False
        if manager.buf.num_sequences < BATCH_SIZE:
            is_random = True
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

            it_count = 0
            while len(cont_env) > 0:
                # make step
                if it_count % 2 == 0:
                    if is_random:
                        actions = np.random.random_sample((total_ranks - 1, 22,))
                    else:
                        actions = manager.predict_actions(observations, training=True)
                it_count += 1

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
                        (*old_observations[cont_env[i] - 1], data['r'], *data['obs'], actions[i].astype('float32')), data['info'])

                    if data['done']:
                        br = True

                if br:
                    manager.on_episode_end(epoch, num_episode)
                    break

            print('Finished episode ', num_episode)
            avg = statistics.mean(rewards)
            logger.log2txt(REWARD_LOGS_PATH,
                           'Epoch: {0}, Ep n: {1}, Avg rew: {2}'.format(epoch, num_episode, avg))

        print('Performing training')
        if not is_random:
            for i in tqdm(range(train_steps)):
                losses = manager.train_step()
                logger.logDict(LOSSES_LOGS_PATH, losses)
        manager.on_epoch_end(epoch)
        print('Buffer length: ', manager.buf.num_sequences, ' sequences')

    for i in range(1, total_ranks):
        comm.send(2, dest=i, tag=11)
else:
    env = MyEnv(visualize=False, memory_size=ENV_MEMORY_SIZE)
    while True:
        data = comm.recv(source=0, tag=11)
        if data == 0:
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

from mpi4py import MPI
import numpy as np
import statistics
import dill

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

MPI.pickle.__init__(dill.dumps, dill.loads)

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
    logger = Logger(100)
    logger.log2txt(DEBUG_LOGS_PATH, 'Loaded: {0} sequences'.format(manager.buf.num_sequences))

    if manager.buf.num_sequences > BATCH_SIZE:
        for i in tqdm(range(PRETRAIN_STEPS)):
            losses = manager.train_step()
            logger.logDict(LOSSES_LOGS_PATH, losses)

    for epoch in range(EPOCHS):
        is_random = manager.buf.num_sequences < BATCH_SIZE

        for num_episode in range(EPISODES_PER_EPOCH):
            manager.on_episode_begin(epoch, num_episode)
            cont_env = [i for i in range(1, total_ranks)]
            cont_env_is_run = [True for i in range(1, total_ranks)]

            rewards = [0.0] * len(cont_env)
            observations = [manager.get_initial_observation() for i in range(total_ranks - 1)]

            for i in range(0, len(cont_env)):
                logger.log_highload(H_DEBUG_LOGS_PATH,
                                    'Sending init message to {0}'.format(cont_env[i]))
                comm.send(0, dest=cont_env[i], tag=11)

            for i in range(0, len(cont_env)):
                logger.log_highload(H_DEBUG_LOGS_PATH,
                                    'Receiving reset messags from {0}'.format(cont_env[i]))
                data = comm.recv(source=cont_env[i], tag=11)
                vf = np.empty((11, 11, 2), dtype=np.float32)
                v = np.empty((97,), dtype=np.float32)
                comm.Recv(vf, source=cont_env[i], tag=13)
                comm.Recv(v, source=cont_env[i], tag=13)
                observations[cont_env[i] - 1] = (vf, v)

            it_count = 0
            logger.log_highload(H_DEBUG_LOGS_PATH,
                                'Starting main loop')
            old_actions = None
            actions = None
            while any(cont_env_is_run) and it_count < MAX_STEPS:
                logger.log_highload(H_DEBUG_LOGS_PATH,
                                    'Predicting actions epoch: {0}, episode: {1}, step: {2}'.format(epoch, num_episode,
                                                                                                    it_count))

                if is_random:
                    actions = np.random.random_sample((total_ranks - 1, 22,))
                else:
                    actions = manager.predict_actions(observations, training=False)
                it_count += N_REPEAT_ACIONS

                old_observations = observations
                observations = [manager.get_initial_observation() for i in range(total_ranks - 1)]

                requests = []
                for i in range(0, len(cont_env)):
                    if cont_env_is_run[i]:
                        req = comm.isend(1, dest=cont_env[i], tag=11)
                        requests.append(req)
                MPI.Request.waitall(requests)

                requests = []
                for i in range(0, len(cont_env)):
                    if cont_env_is_run[i]:
                        req = comm.Isend(actions[i].astype('float32'), dest=cont_env[i], tag=13)
                        requests.append(req)
                MPI.Request.Waitall(requests)

                buffer = [None] * (total_ranks - 1)
                requests = []
                for i in range(0, len(cont_env)):
                    if cont_env_is_run[i]:
                        data = np.empty((3 + ENV_MEMORY_SIZE * (11 * 11 * 2) + ENV_MEMORY_SIZE * 97), dtype=np.float32)
                        buffer[cont_env[i] - 1] = data
                        req = comm.Irecv(buffer[cont_env[i] - 1], source=cont_env[i], tag=13)
                        requests.append(req)
                MPI.Request.Waitall(requests)

                for i in range(0, len(cont_env)):
                    if cont_env_is_run[i]:
                        data = buffer[cont_env[i] - 1]
                        metadata = data[:3]
                        rewards[cont_env[i] - 1] += data[0]
                        vf = np.reshape(data[3: 3 + ENV_MEMORY_SIZE * (11 * 11 * 2)], (11, 11, 2 * ENV_MEMORY_SIZE))
                        v = data[3 + ENV_MEMORY_SIZE * (11 * 11 * 2):]
                        observations[cont_env[i] - 1] = (vf, v)
                        manager.append_observations(
                            (*old_observations[cont_env[i] - 1], data[0], vf, v,
                             actions[i].astype('float32')),
                            int(data[2]))

                        if bool(data[1]):
                            cont_env_is_run[i] = cont_env_is_run[i] and False

            manager.on_episode_end(epoch, num_episode)

            avg = statistics.mean(rewards)
            logger.log2txt(REWARD_LOGS_PATH,
                           'Epoch: {0}, Ep n: {1}, Avg rew: {2}'.format(epoch, num_episode, avg))

        if not is_random:
            for i in tqdm(range(TRAIN_STEPS)):
                losses = manager.train_step()
                logger.logDict(LOSSES_LOGS_PATH, losses)
        manager.on_epoch_end(epoch)
        logger.log2txt(DEBUG_LOGS_PATH,
                       'Buffer length: {0}'.format(manager.buf.num_sequences))

    for i in range(1, total_ranks):
        comm.send(2, dest=i, tag=11)
else:
    env = MyEnv(visualize=False, memory_size=ENV_MEMORY_SIZE, integrator_accuracy=INTEGRATOR_ACCURACY)
    while True:
        data = comm.recv(source=0, tag=11)
        if data == 0:
            observation = env.reset()
            comm.send({
                'r': 0,
                'done': False,
                'info': {'rank': rank}
            }, dest=0, tag=11)
            vf = np.ascontiguousarray(observation[0], dtype=np.float32)
            v = np.ascontiguousarray(observation[1], dtype=np.float32)
            comm.Send(vf, dest=0, tag=13)
            comm.Send(v, dest=0, tag=13)
        elif data == 1:
            action = np.empty(22, dtype=np.float32)
            comm.Recv(action, source=0, tag=13)
            total_reward = 0.0
            done = False
            observation = None
            for i in range(N_REPEAT_ACIONS):
                observation, reward, done, info = env.step(action, obs_as_single_vector=True)
                total_reward += reward
                if done:
                    break
            metadata = np.array([float(total_reward), float(done), float(rank)], dtype=np.float32)
            data = np.concatenate([metadata, observation], axis=-1)
            d = np.ascontiguousarray(data, dtype=np.float32)
            comm.Send(d, dest=0, tag=13)
        elif data == 2:
            break

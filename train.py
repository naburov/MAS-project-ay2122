from mpi4py import MPI
import numpy as np

from MyEnv import MyEnv
from config import *
from replay_buffer import ReplayBuffer

comm = MPI.COMM_WORLD
total_ranks = comm.Get_size()
rank = comm.Get_rank()
env = None
data = None
memory_size = 5
buf = ReplayBuffer(500, memory_size)


def predict_actions(obss):
    return np.zeros((len(obss), 22))


if rank == 0:
    for epoch in range(EPOCHS):
        for num_episode in range(EPISODES_PER_EPOCH):
            # init env
            observations = []
            for i in range(1, total_ranks):
                comm.send(0, dest=i, tag=11)

            for i in range(1, total_ranks):
                data = comm.recv(source=i, tag=11)
                observations.append(data['obs'])

            cont_env = [i for i in range(1, total_ranks)]
            while len(cont_env) > 0:
                # make step
                actions = predict_actions(observations)
                old_observations = observations
                observations = []
                for i in range(0, len(cont_env)):
                    comm.send(1, dest=cont_env[i], tag=11)
                    comm.Send([actions[i], MPI.FLOAT], dest=cont_env[i], tag=77)

                ranks2del = []

                for i in range(0, len(cont_env)):
                    data = comm.recv(source=cont_env[i], tag=11)
                    observations.append(data['obs'])
                    buf.append((*old_observations[i], data['r'], *data['obs']))

                    if data['done']:
                        ranks2del.append(cont_env[i])

                for r in ranks2del:
                    cont_env.remove(r)

            #####
            # do train logic here
            #####
            print('Finished episode ', num_episode)

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

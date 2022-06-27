from mpi4py import MPI
import numpy as np

from MyEnv import MyEnv

comm = MPI.COMM_WORLD
total_ranks = comm.Get_size()
rank = comm.Get_rank()
env = None
data = None
if rank == 0:
    for i in range(1, total_ranks):
        comm.send(0, dest=i, tag=11)

    for i in range(1, total_ranks):
        comm.send(1, dest=i, tag=11)

    for i in range(1, total_ranks):
        data = comm.recv(source=i, tag=11)


    for i in range(1, total_ranks):
        comm.send(2, dest=i, tag=11)
else:
    while True:
        data = comm.recv(source=0, tag=11)
        if data == 0:
            print('init env')
            env = MyEnv(visualize=False)
            obs = env.reset()
        elif data == 1:
            print('making step')
            observation, reward, done, info = env.step(env.action_space.sample())
            comm.send({
                'obs': observation,
                'r': reward,
                'done': done,
                'info': info
            }, dest=0, tag=11)
        elif data == 2:
            print('terminating', data, rank)
            # print(data['a'])
            break

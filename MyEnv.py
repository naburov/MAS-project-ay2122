from osim.env import L2M2019Env
from collections import deque

import numpy as np


def leg_observation_to_vector(leg_obs):
    values = leg_obs['ground_reaction_forces']
    dict_keys = set(leg_obs.keys()) - {'ground_reaction_forces'}
    for key in dict_keys:
        values.extend(leg_obs[key].values())
    return np.array(values)


def observation2tensors(observation):
    tgt_field = observation['v_tgt_field']
    tgt_field = np.transpose(tgt_field, (1, 2, 0))

    pelvis = np.array([
                          observation['pelvis']['height'],
                          observation['pelvis']['pitch'],
                          observation['pelvis']['roll']
                      ] + observation['pelvis']['vel'])

    r_leg = leg_observation_to_vector(observation['r_leg'])
    l_leg = leg_observation_to_vector(observation['l_leg'])
    return tgt_field, np.concatenate((pelvis, r_leg, l_leg), axis=0)


class MyEnv(L2M2019Env):
    def __init__(self, *args, memory_size=5, visualize=True, integrator_accuracy=5e-5, difficulty=3, seed=None,
                 report=None, **kwargs):
        super(MyEnv, self).__init__(visualize=visualize, integrator_accuracy=integrator_accuracy,
                                    difficulty=difficulty, seed=seed,
                                    report=report)
        self.tgt_field_queue = deque(maxlen=memory_size)
        self.body_vector_queue = deque(maxlen=memory_size)
        self.actions_queue = deque(maxlen=memory_size)

    def reset(self, project=True, seed=None, init_pose=None, obs_as_dict=True):
        obs = super(MyEnv, self).reset(project=True, seed=None, init_pose=None, obs_as_dict=True)
        tgt_field, body_vector = observation2tensors(obs)
        for i in range(self.tgt_field_queue.maxlen):
            self.tgt_field_queue.append(tgt_field)
            self.body_vector_queue.append(body_vector)
            self.actions_queue.append(np.full((22), 0.05))
        tgt_field, body_vector = self.flatten_queues()
        return tgt_field, body_vector

    def step(self, action, project=True, obs_as_dict=True):
        obs, reward, done, info = super(MyEnv, self).step(action, project=project, obs_as_dict=obs_as_dict)
        tgt_field, body_vector = observation2tensors(obs)
        self.tgt_field_queue.append(tgt_field)
        self.body_vector_queue.append(body_vector)
        self.actions_queue.append(action)
        tgt_field, body_vector = self.flatten_queues()
        return (tgt_field, body_vector), reward, done, info

    def flatten_queues(self):
        tgt_fields = [f for f in self.tgt_field_queue]
        body_vectors = [f for f in self.body_vector_queue]
        actions = [f for f in self.actions_queue]
        return np.concatenate(tgt_fields, axis=-1), np.concatenate((body_vectors + actions), axis=0)

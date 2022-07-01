import numpy as np
import pickle

TGT_FIELD_SHAPE = (11, 11)
VECTOR_OBS_LEN = 119


class ReplayBuffer:
    def __init__(self, capacity, env_memory_size):
        self.capacity = capacity
        self.counter = 0
        self.vf_state_buffer = np.zeros((capacity, *TGT_FIELD_SHAPE, env_memory_size * 2))
        self.vector_state_bufffer = np.zeros((capacity, VECTOR_OBS_LEN * env_memory_size))
        self.reward_buffer = np.zeros((capacity,))
        self.next_vf_state_buffer = np.zeros((capacity, *TGT_FIELD_SHAPE, env_memory_size * 2))
        self.next_vector_state_bufffer = np.zeros((capacity, VECTOR_OBS_LEN * env_memory_size))

    def append(self, observation):
        index = self.counter % self.capacity
        self.vf_state_buffer[index] = observation[0]
        self.vector_state_bufffer[index] = observation[1]
        self.reward_buffer[index] = observation[2]
        self.next_vf_state_buffer[index] = observation[3]
        self.next_vector_state_bufffer[index] = observation[4]
        self.counter += 1

    def sample(self, batch_size):
        if batch_size > self.counter:
            batch_size = self.counter
        indices = np.random.choice(self.counter, batch_size)
        return (
            self.vf_state_buffer[indices],
            self.vector_state_bufffer[indices],
            self.reward_buffer[indices],
            self.next_vf_state_buffer[indices],
            self.next_vector_state_bufffer[indices]
        )

    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

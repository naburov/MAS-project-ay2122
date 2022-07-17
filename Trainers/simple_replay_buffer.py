import numpy as np
import tensorflow as tf
import pickle

TGT_FIELD_SHAPE = (11, 11)
VECTOR_OBS_LEN = 119
NUM_ACTIONS = 22


class SimpleReplayBuffer:
    def __init__(self, capacity, env_memory_size):
        self.capacity = capacity
        self.counter = 0
        self.vf_state_buffer = np.zeros((capacity, *TGT_FIELD_SHAPE, env_memory_size * 2))
        self.vector_state_bufffer = np.zeros((capacity, VECTOR_OBS_LEN * env_memory_size))
        self.action_buffer = np.zeros((capacity, NUM_ACTIONS))
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
        self.action_buffer[index] = observation[5]
        self.counter += 1

    def sample(self, batch_size, convert_to_tf_tensor=True):
        if batch_size > self.counter:
            batch_size = self.counter
        indices = np.random.choice(min(self.counter, self.capacity), batch_size)
        if not convert_to_tf_tensor:
            return (
                self.vf_state_buffer[indices],
                self.vector_state_bufffer[indices],
                self.reward_buffer[indices],
                self.next_vf_state_buffer[indices],
                self.next_vector_state_bufffer[indices],
                self.action_buffer[indices]
            )
        else:
            return (
                tf.convert_to_tensor(
                    self.vf_state_buffer[indices], dtype=tf.float32),
                tf.convert_to_tensor(
                    self.vector_state_bufffer[indices], dtype=tf.float32),
                tf.convert_to_tensor(
                    self.reward_buffer[indices], dtype=tf.float32),
                tf.convert_to_tensor(
                    self.next_vf_state_buffer[indices], dtype=tf.float32),
                tf.convert_to_tensor(
                    self.next_vector_state_bufffer[indices], dtype=tf.float32),
                tf.convert_to_tensor(
                    self.action_buffer[indices], dtype=tf.float32)
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

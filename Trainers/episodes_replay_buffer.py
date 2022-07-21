import numpy as np
import tensorflow as tf
import pickle

TGT_FIELD_SHAPE = (11, 11)
VECTOR_OBS_LEN = 119
NUM_ACTIONS = 22


class EpisodeReplayBuffer:
    def __init__(self, seq_capacity, num_ranks, env_memory_size):
        self.capacity = (seq_capacity // (num_ranks - 1) + 1) * (num_ranks - 1)
        self.sequences = [None] * self.capacity
        self.num_sequences = 0
        self.active_indices = [i for i in range(num_ranks - 1)]
        self.env_memory_size = env_memory_size

    @property
    def current_len(self):
        return sum([len(s) for s in self.sequences if s is not None])

    def prepare_buffers(self, num_ranks):
        if self.active_indices[-1] == self.capacity - 1:
            self.active_indices = [i for i in range(num_ranks - 1)]
        else:
            self.active_indices = [i + num_ranks - 1 for i in self.active_indices]

        for id in self.active_indices:
            self.sequences[id] = None

        self.num_sequences += max(0, min(self.num_sequences + num_ranks - 1, self.capacity - 1))

    def append(self, observation, info):
        # print(len(self.sequences), info['rank'] - 1, self.active_indices, self.capacity)
        if self.sequences[self.active_indices[info['rank'] - 1]] is None:
            self.sequences[self.active_indices[info['rank'] - 1]] = [observation]
        else:
            self.sequences[self.active_indices[info['rank'] - 1]].append(observation)

    def sample_sequences(self, num_sequences):
        if num_sequences > self.num_sequences:
            num_sequences = self.num_sequences
        indices = np.random.choice(min(self.num_sequences, self.capacity), num_sequences)
        return [self.sequences[i] for i in indices]

    def sample(self, batch_size, convert_to_tf__tensors=True):
        if self.current_len < batch_size:
            batch_size = self.current_len

        indices = sorted(np.random.choice(self.current_len, batch_size, replace=False))
        last = 0

        vf_state_buffer = np.zeros((batch_size, *TGT_FIELD_SHAPE, self.env_memory_size * 2))
        vector_state_bufffer = np.zeros((batch_size, VECTOR_OBS_LEN * self.env_memory_size))
        action_buffer = np.zeros((batch_size, NUM_ACTIONS))
        reward_buffer = np.zeros((batch_size,))
        next_vf_state_buffer = np.zeros((batch_size, *TGT_FIELD_SHAPE, self.env_memory_size * 2))
        next_vector_state_bufffer = np.zeros((batch_size, VECTOR_OBS_LEN * self.env_memory_size))
        counter = 0
        for s in self.sequences:
            if s is None:
                break
            i = 0
            while last < batch_size and i < len(s):
                if counter == indices[last]:
                    o = s[i]
                    vf_state_buffer[last] = o[0]
                    vector_state_bufffer[last] = o[1]
                    reward_buffer[last] = o[2]
                    next_vf_state_buffer[last] = o[3]
                    next_vector_state_bufffer[last] = o[4]
                    action_buffer[last] = o[5]
                    last += 1
                counter += 1
                i += 1

        if not convert_to_tf__tensors:
            return (
                vf_state_buffer,
                vector_state_bufffer,
                reward_buffer,
                next_vf_state_buffer,
                next_vector_state_bufffer,
                action_buffer
            )
        else:
            return (
                tf.convert_to_tensor(
                    vf_state_buffer, dtype=tf.float32),
                tf.convert_to_tensor(
                    vector_state_bufffer, dtype=tf.float32),
                tf.convert_to_tensor(
                    reward_buffer, dtype=tf.float32),
                tf.convert_to_tensor(
                    next_vf_state_buffer, dtype=tf.float32),
                tf.convert_to_tensor(
                    next_vector_state_bufffer, dtype=tf.float32),
                tf.convert_to_tensor(
                    action_buffer, dtype=tf.float32)
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

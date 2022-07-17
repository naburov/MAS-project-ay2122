import numpy as np
import tensorflow as tf
import pickle

TGT_FIELD_SHAPE = (11, 11)
VECTOR_OBS_LEN = 119
NUM_ACTIONS = 22


class EpisodeReplayBuffer:
    def __init__(self, seq_capacity, num_ranks, env_memory_size):
        self.sequences = []
        self.capacity = ((seq_capacity // num_ranks) + 1) * num_ranks
        self.seq_counter = 0
        self.num_sequences = 0
        self.active_indices = [i for i in range(num_ranks)]
        self.env_memory_size = env_memory_size

    @property
    def current_len(self):
        return sum([len(s) for s in self.sequences])

    def prepare_buffers(self, num_ranks):
        if self.seq_counter == self.capacity:
            self.active_indices = [i for i in range(num_ranks)]
        else:
            self.active_indices = [i + num_ranks for i in self.active_indices]

        self.num_sequences += max(0, min(self.num_sequences + num_ranks, len(self.capacity) - 1))

    def append(self, observation, info):
        print(len(self.sequences), -info['rank'])
        self.sequences[self.active_indices[-info['rank']]].append(observation)

    def sample_sequences(self, num_sequences):
        if num_sequences > self.num_sequences:
            num_sequences = self.num_sequences
        indices = np.random.choice(min(self.num_sequences, self.capacity), num_sequences)
        return self.sequences[indices]

    def sample(self, batch_size, convert_to_tf__tensors=True):
        if self.current_len < batch_size:
            batch_size = self.current_len

        indices = np.random.choice(self.current_len, batch_size)
        last = 0

        vf_state_buffer = np.zeros((batch_size, *TGT_FIELD_SHAPE, self.env_memory_size * 2))
        vector_state_bufffer = np.zeros((batch_size, VECTOR_OBS_LEN * self.env_memory_size))
        action_buffer = np.zeros((batch_size, NUM_ACTIONS))
        reward_buffer = np.zeros((batch_size,))
        next_vf_state_buffer = np.zeros((batch_size, *TGT_FIELD_SHAPE, self.env_memory_size * 2))
        next_vector_state_bufffer = np.zeros((batch_size, VECTOR_OBS_LEN * self.env_memory_size))

        counter = 0
        for s in self.sequences:
            if counter + len(s) < indices[last]:
                counter += len(s)
                continue
            for o in s:
                if counter == indices[last]:
                    vf_state_buffer[last] = o[0]
                    vector_state_bufffer[last] = o[1]
                    reward_buffer[last] = o[2]
                    next_vf_state_buffer[last] = o[3]
                    next_vector_state_bufffer[last] = o[4]
                    action_buffer[last] = o[5]
                    last += 1
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

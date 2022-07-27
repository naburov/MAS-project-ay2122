import random

import numpy as np
import tensorflow as tf
import pickle

TGT_FIELD_SHAPE = (11, 11)
VECTOR_OBS_LEN = 97
NUM_ACTIONS = 22


class EpisodeReplayBuffer:
    def __init__(self, seq_capacity, num_ranks, env_memory_size):
        self.capacity = int(np.ceil(seq_capacity / (num_ranks - 1))) * (num_ranks - 1)
        self.sequences = [None] * self.capacity
        self.num_sequences = 0
        self.active_indices = [self.capacity - 1]
        self.env_memory_size = env_memory_size

    @property
    def current_len(self):
        return sum([len(s) for s in self.sequences if s is not None])

    def prepare_buffers(self, num_ranks):
        if self.active_indices[-1] == self.capacity - 1:
            self.active_indices = [i for i in range(num_ranks - 1)]
        else:
            self.active_indices = [self.active_indices[-1] + i for i in range(1, num_ranks)]

        for id in self.active_indices:
            self.sequences[id] = None

    def finish_episode(self, num_ranks):
        self.num_sequences = max(0, min(self.num_sequences + num_ranks - 1, self.capacity))

    def append(self, observation, info):
        if self.sequences[self.active_indices[info['rank'] - 1]] is None:
            self.sequences[self.active_indices[info['rank'] - 1]] = [observation]
        else:
            self.sequences[self.active_indices[info['rank'] - 1]].append(observation)

    def sample_sequences(self, num_sequences):
        if num_sequences > self.num_sequences:
            num_sequences = self.num_sequences
        indices = np.random.choice(min(self.num_sequences, self.capacity), num_sequences)
        return [self.sequences[i] for i in indices]

    def sample_sequences_tensors(self, num_sequences, n_steps, convert_to_tf__tensors=False):
        vf_state_buffer = np.zeros((n_steps, num_sequences, *TGT_FIELD_SHAPE, self.env_memory_size * 2))
        vector_state_bufffer = np.zeros((n_steps, num_sequences, VECTOR_OBS_LEN * self.env_memory_size))
        action_buffer = np.zeros((n_steps, num_sequences, NUM_ACTIONS))
        reward_buffer = np.zeros((n_steps, num_sequences,))
        next_vf_state_buffer = np.zeros((n_steps, num_sequences, *TGT_FIELD_SHAPE, self.env_memory_size * 2))
        next_vector_state_bufffer = np.zeros((n_steps, num_sequences, VECTOR_OBS_LEN * self.env_memory_size))

        if num_sequences > self.num_sequences:
            num_sequences = self.num_sequences

        indices = np.random.choice(min(self.num_sequences - 1, self.capacity), num_sequences)

        for i in range(len(indices)):
            start = random.randint(0, len(self.sequences[indices[i]]) - n_steps)
            for j in range(start, start + n_steps):
                o = self.sequences[indices[i]][j]
                vf_state_buffer[j - start, i] = o[0]
                vector_state_bufffer[j - start, i] = o[1]
                reward_buffer[j - start, i] = o[2]
                next_vf_state_buffer[j - start, i] = o[3]
                next_vector_state_bufffer[j - start, i] = o[4]
                action_buffer[j - start, i] = o[5]

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

    def save_sequences(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.sequences, f)

    def load_sequences(self, filename):
        with open(filename, 'rb') as f:
            self.sequences = pickle.load(f)
        if len(self.sequences) < self.capacity:
            self.sequences.extend([None] * (self.capacity - len(self.sequences)))
        elif len(self.sequences) > self.capacity:
            self.sequences = self.sequences[:self.capacity]
        self.num_sequences = sum([1 for s in self.sequences if s is not None])
        self.active_indices = [self.num_sequences - 1]

import random

import numpy as np
import tensorflow as tf
import pickle


# TGT_FIELD_SHAPE = (11, 11)
# VECTOR_OBS_LEN = 119
# NUM_ACTIONS = 22


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

        # print('Active indices ', self.active_indices, ' Num sequences ', self.num_sequences)

    def finish_episode(self, num_ranks):
        self.num_sequences = max(0, min(self.num_sequences + num_ranks - 1, self.capacity))

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

    def sample_sequences_tensors(self, num_sequences, n_steps, convert_to_tf__tensors=False):
        state_buffer = np.zeros((n_steps, num_sequences, 3))
        action_buffer = np.zeros((n_steps, num_sequences, 1))
        reward_buffer = np.zeros((n_steps, num_sequences,))
        next_state_buffer = np.zeros((n_steps, num_sequences, 3))
        if num_sequences > self.num_sequences:
            num_sequences = self.num_sequences
        indices = np.random.choice(min(self.num_sequences - 1, self.capacity), num_sequences)
        for i in range(len(indices)):
            start = random.randint(0, len(self.sequences[indices[i]]) - n_steps)
            # old_observations, reward, observation, actions[0]
            for j in range(start, start + n_steps):
                o = self.sequences[indices[i]][j]
                state_buffer[j - start, i] = o[0]
                reward_buffer[j - start, i] = o[1]
                next_state_buffer[j - start, i] = o[2]
                action_buffer[j - start, i] = o[3]
        if not convert_to_tf__tensors:
            return (
                state_buffer,
                reward_buffer,
                next_state_buffer,
                action_buffer
            )
        else:
            return (
                tf.convert_to_tensor(
                    state_buffer, dtype=tf.float32),
                tf.convert_to_tensor(
                    reward_buffer, dtype=tf.float32),
                tf.convert_to_tensor(
                    next_state_buffer, dtype=tf.float32),
                tf.convert_to_tensor(
                    action_buffer, dtype=tf.float32)
            )

    def sample(self, batch_size, convert_to_tf__tensors=True):
        if self.current_len < batch_size:
            batch_size = self.current_len

        indices = sorted(np.random.choice(self.current_len, batch_size, replace=False))
        last = 0
        state_buffer = np.zeros((batch_size, 3))
        action_buffer = np.zeros((batch_size, 1))
        reward_buffer = np.zeros((batch_size,))
        next_state_buffer = np.zeros((batch_size, 3))
        counter = 0
        for s in self.sequences:
            if s is None:
                break
            i = 0
            while last < batch_size and i < len(s):
                if counter == indices[last]:
                    o = s[i]
                    state_buffer[last] = o[0]
                    reward_buffer[last] = o[1]
                    next_state_buffer[last] = o[2]
                    action_buffer[last] = o[3]
                    last += 1
                counter += 1
                i += 1

        if not convert_to_tf__tensors:
            return (
                state_buffer,
                reward_buffer,
                next_state_buffer,
                action_buffer
            )
        else:
            return (
                tf.convert_to_tensor(
                    state_buffer, dtype=tf.float32),
                tf.convert_to_tensor(
                    reward_buffer, dtype=tf.float32),
                tf.convert_to_tensor(
                    next_state_buffer, dtype=tf.float32),
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

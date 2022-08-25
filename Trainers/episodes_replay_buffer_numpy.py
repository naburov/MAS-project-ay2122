import random
import glob
import os

import numpy as np
import tensorflow as tf
import pickle

TGT_FIELD_SHAPE = (11, 11)
VECTOR_OBS_LEN = 97
NUM_ACTIONS = 22


class EpisodeReplayBufferNumpy:
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
        for id in self.active_indices:
            res_tensors = []
            for i in range(len(self.sequences[id][0])):
                res_tensors.append(
                    np.stack([
                        self.sequences[id][j][i] for j in range(len(self.sequences[id]))
                    ], axis=0)
                )
            self.sequences[id] = res_tensors

    def append(self, observation, rank):
        if self.sequences[self.active_indices[rank - 1]] is None:
            self.sequences[self.active_indices[rank - 1]] = [observation]
        else:
            self.sequences[self.active_indices[rank - 1]].append(observation)

    def sample_sequences(self, num_sequences):
        if num_sequences > self.num_sequences:
            num_sequences = self.num_sequences
        indices = np.random.choice(min(self.num_sequences, self.capacity), num_sequences)
        return [self.sequences[i] for i in indices]

    def sample_sequences_tensors(self, num_sequences, n_steps, convert_to_tf_tensors=False):
        if num_sequences > self.num_sequences:
            num_sequences = self.num_sequences

        indices = np.random.choice(min(self.num_sequences - 1, self.capacity), num_sequences)

        batch = [np.zeros((num_sequences, n_steps, *t.shape[1:])) for t in self.sequences[0]]
        for i in range(len(indices)):
            start = random.randint(0, self.sequences[indices[i]][0].shape[0] - n_steps)
            for j in range(len(self.sequences[0])):
                batch[j][i] = self.sequences[indices[i]][j][start: start + n_steps]

        batch = [np.transpose(t, (1, 0, *tuple(list(range(2, len(t.shape)))))) for t in batch]

        if not convert_to_tf_tensors:
            return tuple(batch)
        else:
            return tuple([tf.convert_to_tensor(t, dtype=tf.float32) for t in batch])

    def save_array(self, fname, arr):
        with open(fname, 'wb') as f:
            np.save(f, arr)

    def load_array(self, fname):
        with open(fname, 'rb') as f:
            arr = np.load(f)
        return arr

    def save_sequences(self, save_dir):
        for i in range(len(self.sequences)):
            if not self.sequences[i] is None:
                for j in range(len(self.sequences[i])):
                    fname = os.path.join(save_dir, '_'.join([str(i), str(j)]) + '.npy')
                    self.save_array(fname, self.sequences[i][j])

    def load_sequences(self, load_dir):
        fnames = glob.glob(
            os.path.join(load_dir, '*')
        )
        self.sequences = [None] * (len(fnames) // 6)
        for name in fnames:
            ids = name.split(os.sep)[-1].split('.')[0].split('_')
            ids = [int(id) for id in ids]
            if self.sequences[ids[0]] is None:
                self.sequences[ids[0]] = [None] * 6
            self.sequences[ids[0]][ids[1]] = self.load_array(name)

        if len(self.sequences) < self.capacity:
            self.sequences.extend([None] * (self.capacity - len(self.sequences)))
        elif len(self.sequences) > self.capacity:
            self.sequences = self.sequences[:self.capacity]
        self.num_sequences = sum([1 for s in self.sequences if s is not None])
        self.active_indices = [self.num_sequences - 1]

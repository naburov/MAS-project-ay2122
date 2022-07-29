from Trainers.Dreamer.DreamerV1 import Dreamer
from Trainers.episodes_replay_buffer import EpisodeReplayBuffer as ReplayBuffer
from Trainers.Trainer import TrainManager
import os
import numpy as np
from config import *


class DreamerTrainManager(TrainManager):
    def __init__(self, checkpoint_dir, buffer_path, memory_size, buffer_capacity, num_ranks, batch_size=32):
        super().__init__()
        self.dreamer = Dreamer(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        self.buffer_path = buffer_path
        self.batch_size = batch_size
        self.num_ranks = num_ranks
        self.sequence_length = SEQUENCE_LENGTH

        self.buf = ReplayBuffer(buffer_capacity, num_ranks, memory_size)

        if os.path.exists(self.buffer_path):
            self.buf.load_sequences(self.buffer_path)

        self.prev_actions = None
        self.prev_state = None

    def predict_actions(self, obss, training):
        vfs = np.stack([o[0] for o in obss], axis=0).astype('float32')
        vs = np.stack([o[1] for o in obss], axis=0).astype('float32')
        action, state = self.dreamer.policy((vfs, vs), self.prev_state, self.prev_actions, training=training)
        self.prev_actions = action
        self.prev_state = state
        return action.numpy().astype('float64')

    def train_step(self):
        seq = self.buf.sample_sequences_tensors(self.batch_size, self.sequence_length, True)
        return self.dreamer.train_step(observations=(seq[0], seq[1]), rewards=seq[2], actions=seq[-1])

    def on_episode_begin(self, epoch_n, episode_n):
        self.buf.prepare_buffers(self.num_ranks)
        self.prev_actions = None
        self.prev_state = None

    def on_episode_end(self, epoch_n, episode_n):
        self.buf.finish_episode(self.num_ranks)

    def on_epoch_end(self, epoch_n):
        self.dreamer.save_state(self.checkpoint_dir)
        if epoch_n % 20 == 0:
            self.buf.save_sequences(self.buffer_path)

    def append_observations(self, data, info):
        self.buf.append(data, info)

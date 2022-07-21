from Trainers.DDPGTrainManager.models import get_actor_model, get_critic_model
from Trainers.DDPGTrainManager.noise_utils import OUActionNoise
# from Trainers.simple_replay_buffer import SimpleReplayBuffer as ReplayBuffer
from Trainers.Dreamer.DreamerV1 import Dreamer
from Trainers.episodes_replay_buffer import EpisodeReplayBuffer as ReplayBuffer
from Trainers.DDPGTrainManager.train_utils import train_step, update_target
from Trainers.Trainer import TrainManager
import os
import numpy as np

tau = 0.01
std_dev = 0.2


class DreamerTrainManager(TrainManager):
    def __init__(self, checkpoint_dir, buffer_path, memory_size, buffer_capacity, num_ranks, batch_size=32):
        super().__init__()
        self.dreamer = Dreamer(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        self.buffer_path = buffer_path
        self.batch_size = 3
        self.num_ranks = num_ranks

        self.buf = ReplayBuffer(buffer_capacity, num_ranks, memory_size)

        if os.path.exists(self.buffer_path):
            self.buf.load(self.buffer_path)

        self.prev_actions = None
        self.prev_state = None

    def predict_actions(self, obss):
        vfs = np.stack([o[0] for o in obss], axis=0)
        vs = np.stack([o[1] for o in obss], axis=0)
        action, state = self.dreamer.policy((vfs, vs), self.prev_state, self.prev_actions)
        self.prev_actions = action
        self.prev_state = state
        return action

    def train_step(self):
        # vf, v, r, next_vf, next_v, a = self.buf.sample_sequences(self.batch_size)
        seq = self.buf.sample_sequences(self.batch_size)
        res_seq = []
        for s in seq:
            vf = np.stack([record[0] for record in s])
            v = np.stack([record[1] for record in s])
            a = np.stack([record[-1] for record in s])
            r = np.stack([record[2] for record in s])
            res_seq.append(((vf, v), a, r))
        self.dreamer.train_step(res_seq)

    def on_episode_begin(self, epoch_n, episode_n):
        self.buf.prepare_buffers(self.num_ranks)

    def on_epoch_end(self, epoch_n):
        self.dreamer.save_state(self.checkpoint_dir)
        if epoch_n % 10 == 0:
            self.buf.save(self.buffer_path)

    def append_observations(self, data, info):
        self.buf.append(data, info)

from Trainers.DDPGTrainManager.models import get_actor_model, get_critic_model
from Trainers.DDPGTrainManager.noise_utils import OUActionNoise
# from Trainers.simple_replay_buffer import SimpleReplayBuffer as ReplayBuffer
from Trainers.episodes_replay_buffer import EpisodeReplayBuffer as ReplayBuffer
from Trainers.DDPGTrainManager.train_utils import train_step, update_target
from Trainers.Trainer import TrainManager
import os
import numpy as np

tau = 0.01
std_dev = 0.2


class DDPGTrainManager(TrainManager):
    def __init__(self, checkpoint_dir, buffer_path, memory_size, buffer_capacity, num_ranks, batch_size=32):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.buffer_path = buffer_path
        self.batch_size = batch_size
        self.num_ranks = num_ranks

        self.actor_model = get_actor_model(memory_size)
        self.critic_model = get_critic_model(memory_size)

        self.target_actor = get_actor_model(memory_size)
        self.target_critic = get_critic_model(memory_size)

        if os.path.exists(os.path.join(checkpoint_dir, 'actor.h5')):
            self.actor_model.load_weights(os.path.join(checkpoint_dir, 'actor.h5'))
            self.critic_model.load_weights(os.path.join(checkpoint_dir, 'critic_model.h5'))

            self.target_actor.load_weights(os.path.join(checkpoint_dir, 'target_actor.h5'))
            self.target_critic.load_weights(os.path.join(checkpoint_dir, 'target_critic.h5'))

        self.buf = ReplayBuffer(buffer_capacity, num_ranks, memory_size)
        self.ou_noise = OUActionNoise(mean=np.full((22,), 0.5), std_deviation=0.5 * np.ones(1))
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())
        if os.path.exists(self.buffer_path):
            self.buf.load(self.buffer_path)

    def predict_actions(self, obss):
        vfs = np.stack([o[0] for o in obss], axis=0)
        vs = np.stack([o[1] for o in obss], axis=0)
        noise = self.ou_noise.sample_batch(len(obss))
        pred_actions = self.actor_model((vfs, vs)).numpy() + noise
        legal_action = np.clip(pred_actions, 0., 1.)
        return legal_action

    def train_step(self):
        train_step(
            self.batch_size,
            self.buf,
            self.target_actor,
            self.target_critic,
            self.critic_model,
            self.actor_model
        )
        update_target(self.target_actor.variables, self.actor_model.variables, tau)
        update_target(self.target_critic.variables, self.critic_model.variables, tau)

        self.actor_model.save_weights(os.path.join(self.checkpoint_dir, 'actor.h5'))
        self.critic_model.save_weights(os.path.join(self.checkpoint_dir, 'critic_model.h5'))

        self.target_actor.save_weights(os.path.join(self.checkpoint_dir, 'target_actor.h5'))
        self.target_critic.save_weights(os.path.join(self.checkpoint_dir, 'target_critic.h5'))

    def on_episode_begin(self, epoch_n, episode_n):
        self.buf.prepare_buffers(self.num_ranks)

    def on_epoch_end(self, epoch_n):
        self.actor_model.save_weights(os.path.join(self.checkpoint_dir, 'actor.h5'))
        self.critic_model.save_weights(os.path.join(self.checkpoint_dir, 'critic_model.h5'))

        self.target_actor.save_weights(os.path.join(self.checkpoint_dir, 'target_actor.h5'))
        self.target_critic.save_weights(os.path.join(self.checkpoint_dir, 'target_critic.h5'))
        if epoch_n % 10 == 0:
            self.buf.save(self.buffer_path)

    def append_observations(self, data, info):
        self.buf.append(data, info)

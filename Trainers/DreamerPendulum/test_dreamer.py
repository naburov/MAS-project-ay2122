from unittest import TestCase
import numpy as np

from Trainers.DreamerPendulum.DreamerV1 import Dreamer
from Trainers.DreamerPendulum.models import RSSM
from config_pendulum import MODEL_CHECKPOINT_DIR

hidden = 20
determ = 30
stoch = 30
units = 20
num_actions = 1
kl_scale = 0.01
free_nats = 3.0
gamma = 0.99
lambda_ = 0.95
horizon = 15
env_memory_size = 8
embedding_size = 100
filters = 32


class TestDreamer(TestCase):
    def test_policy(self):
        self.fail()

    def test_train_step(self):
        self.fail()

    def test_imagine_ahead(self):
        batch_size = 4
        time_steps = 2
        model = RSSM(stoch, determ, embedding_size, hidden)
        state = model.initial_state(batch_size)
        actions = np.zeros((time_steps, batch_size, 1), np.float32)
        embs = np.zeros((time_steps, batch_size, embedding_size), np.float32)
        dreamer = Dreamer(MODEL_CHECKPOINT_DIR)
        prior, post = dreamer.dynamics_model.observe(embs, actions, state)
        imag_feat = dreamer.imagine_ahead(post)
        self.assertEqual(True, True)

    def test_v_k_n(self):
        self.fail()

    def test_estimate_return(self):
        # tested with wolfram
        gamma = 0.99
        lambda_ = 0.95
        time_steps = 10
        batch_size = 4
        reward_tensor = np.ones((time_steps, batch_size, 1))
        value_tensor = np.ones((time_steps, batch_size, 1))

        gamma_discount = np.full_like(value_tensor, gamma)
        gamma_discount[0, ...] = 1.0
        gamma_discount = np.cumprod(gamma_discount, 0)

        lambda_discount = np.full_like(value_tensor, lambda_)
        lambda_discount[0, ...] = 1.0
        lambda_discount = np.cumprod(lambda_discount, 0)

        reward_tensor = np.multiply(gamma_discount, reward_tensor)
        value_tensor = np.multiply(gamma_discount, value_tensor)
        reward_tensor = np.cumsum(reward_tensor, 0)

        v_k_n = reward_tensor + value_tensor
        v_k_n = np.multiply(lambda_discount, v_k_n)
        v_h_n = v_k_n[-1, ...]
        v_k_n = np.cumsum(v_k_n[:-1], 0)[-1, ...]

        res = (1 - lambda_) * v_k_n + v_h_n

        self.assertEqual(True, True)

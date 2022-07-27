from unittest import TestCase
import numpy as np

from Trainers.DreamerPendulum.models import RSSM

units = 100
hidden = 200
determenistic = 200
stochastic = 100
emb_shape = 50

NUM_ACTIONS = 1


class TestRSSM(TestCase):
    def assertStateShape(self, state, batch_size):
        self.assertEqual(state.deter.shape, (batch_size, determenistic))
        self.assertEqual(state.mean.shape, (batch_size, stochastic))
        self.assertEqual(state.std.shape, (batch_size, stochastic))
        self.assertEqual(state.stoch.shape, (batch_size, stochastic))

    def test_initial_state(self):
        batch_size = 4
        model = RSSM(stochastic, determenistic, emb_shape, hidden)
        init_state = model.initial_state(batch_size)
        self.assertStateShape(init_state, batch_size)

    def test_observe(self):
        batch_size = 4
        time_steps = 2
        model = RSSM(stochastic, determenistic, emb_shape, hidden)
        state = model.initial_state(batch_size)
        actions = np.zeros((time_steps, batch_size, NUM_ACTIONS), np.float32)
        embs = np.zeros((time_steps, batch_size, emb_shape), np.float32)
        prior, post = model.observe(embs, actions, state)

        self.assertEqual(prior.deter.shape, (time_steps, batch_size, determenistic))
        self.assertEqual(prior.mean.shape, (time_steps, batch_size, stochastic))
        self.assertEqual(prior.std.shape, (time_steps, batch_size, stochastic))
        self.assertEqual(prior.stoch.shape, (time_steps, batch_size, stochastic))

        for f in prior:
            self.assertEqual(np.isnan(f.numpy()).any(), False)
        for f in post:
            self.assertEqual(np.isnan(f.numpy()).any(), False)

    def test_posterior(self):
        batch_size = 4
        model = RSSM(stochastic, determenistic, emb_shape, hidden)
        state = model.initial_state(batch_size)
        actions = np.zeros((batch_size, NUM_ACTIONS), np.float32)
        embs = np.zeros((batch_size, emb_shape), np.float32)
        prior, post = model.posterior(state, actions, embs)

        for f in prior:
            self.assertEqual(np.isnan(f.numpy()).any(), False)
        for f in post:
            self.assertEqual(np.isnan(f.numpy()).any(), False)

    def test_prior(self):
        batch_size = 4
        model = RSSM(stochastic, determenistic, emb_shape, hidden)
        state = model.initial_state(batch_size)
        actions = np.zeros((batch_size, NUM_ACTIONS), np.float32)
        prior = model.prior(state, actions)
        self.assertStateShape(prior, batch_size)

        for f in prior:
            self.assertEqual(np.isnan(f.numpy()).any(), False)

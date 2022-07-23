import pickle
from unittest import TestCase

from Trainers.episodes_replay_buffer import EpisodeReplayBuffer


class TestEpisodeReplayBuffer(TestCase):

    def test_init_not_equal(self):
        seq_capacity = 10
        num_ranks = 4
        env_memory_size = 5
        self.buf = EpisodeReplayBuffer(seq_capacity, num_ranks, env_memory_size)
        self.assertEqual(self.buf.capacity, 12)

    def test_init_equal(self):
        seq_capacity = 12
        num_ranks = 4
        env_memory_size = 5
        self.buf = EpisodeReplayBuffer(seq_capacity, num_ranks, env_memory_size)
        self.assertEqual(self.buf.capacity, 12)

    def test_sample_sequences_tensor(self):
        with open(r'C:\Users\burov\Projects\mas-project-burov-ay2122\test_seq_list.pkl', 'rb') as f:
            test_sequences = pickle.load(f)
        seq_capacity = 12
        num_ranks = 4
        env_memory_size = 8
        buf = EpisodeReplayBuffer(seq_capacity, num_ranks, env_memory_size)
        buf.prepare_buffers(num_ranks)
        cur_len = 0
        for i in range(min(buf.capacity, len(test_sequences))):
            cur_len += len(test_sequences[i])
            for obs in test_sequences[i]:
                o, info = obs
                buf.append(o, {'rank': i % (num_ranks - 1) + 1})

        sampled = buf.sample_sequences_tensors(3, 15)
        self.assertEqual(sampled[-1].shape, (15, 3, 22))

    def test_current_len_single_thread(self):
        with open(r'C:\Users\burov\Projects\mas-project-burov-ay2122\test_seq_list.pkl', 'rb') as f:
            test_sequences = pickle.load(f)
        seq_capacity = 12
        num_ranks = 2
        env_memory_size = 8
        buf = EpisodeReplayBuffer(seq_capacity, num_ranks, env_memory_size)
        buf.prepare_buffers(num_ranks)
        cur_len = 0
        for s in test_sequences[:buf.capacity]:
            cur_len += len(s)
            for obs in s:
                o, info = obs
                buf.append(o, info)
        print(buf.current_len, cur_len)
        self.assertEqual(buf.current_len, cur_len)

    def test_current_len_multithread(self):
        with open(r'C:\Users\burov\Projects\mas-project-burov-ay2122\test_seq_list.pkl', 'rb') as f:
            test_sequences = pickle.load(f)
        seq_capacity = 12
        num_ranks = 4
        env_memory_size = 8
        buf = EpisodeReplayBuffer(seq_capacity, num_ranks, env_memory_size)
        buf.prepare_buffers(num_ranks)
        cur_len = 0
        for i in range(min(buf.capacity, len(test_sequences))):
            cur_len += len(test_sequences[i])
            for obs in test_sequences[i]:
                o, info = obs
                buf.append(o, {'rank': i % (num_ranks - 1) + 1})
        self.assertEqual(buf.current_len, cur_len)

    def test_prepare_buffers_begin(self):
        seq_capacity = 10
        num_ranks = 4
        env_memory_size = 5
        self.buf = EpisodeReplayBuffer(seq_capacity, num_ranks, env_memory_size)
        self.buf.prepare_buffers(num_ranks)
        for i in range(num_ranks - 1):
            self.assertEqual(self.buf.active_indices[i], i)
        self.assertEqual(len(self.buf.active_indices), num_ranks - 1)
        self.assertEqual(self.buf.num_sequences, num_ranks - 1)

    def test_prepare_buffers_middle(self):
        seq_capacity = 10
        num_ranks = 4
        env_memory_size = 5
        self.buf = EpisodeReplayBuffer(seq_capacity, num_ranks, env_memory_size)
        self.buf.prepare_buffers(num_ranks)
        self.buf.prepare_buffers(num_ranks)
        for i in range(num_ranks - 1):
            self.assertEqual(self.buf.active_indices[i], i + num_ranks - 1)
        self.assertEqual(self.buf.num_sequences, (num_ranks - 1) * 2)

    def test_prepare_buffers_end(self):
        seq_capacity = 10
        num_ranks = 4
        env_memory_size = 5
        n_add = 3
        self.buf = EpisodeReplayBuffer(seq_capacity, num_ranks, env_memory_size)
        for j in range(n_add + 1):
            self.buf.prepare_buffers(num_ranks)

        for i in range(num_ranks - 1):
            self.assertEqual(self.buf.active_indices[i], i + (num_ranks - 1) * n_add)
        self.assertEqual(self.buf.num_sequences, 12)

    def test_prepare_buffers_repeat(self):
        seq_capacity = 10
        num_ranks = 4
        env_memory_size = 5
        n_add = 3
        self.buf = EpisodeReplayBuffer(seq_capacity, num_ranks, env_memory_size)
        for j in range(n_add + 2):
            self.buf.prepare_buffers(num_ranks)

        for i in range(num_ranks - 1):
            self.assertEqual(self.buf.active_indices[i], i)
        self.assertEqual(self.buf.num_sequences, 12)
        self.assertEqual(self.buf.num_sequences, self.buf.capacity)

    def test_sample_sequences_less(self):
        with open(r'C:\Users\burov\Projects\mas-project-burov-ay2122\test_seq_list.pkl', 'rb') as f:
            test_sequences = pickle.load(f)
        seq_capacity = 12
        num_ranks = 4
        env_memory_size = 8
        to_sample = 3
        buf = EpisodeReplayBuffer(seq_capacity, num_ranks, env_memory_size)
        buf.prepare_buffers(num_ranks)
        cur_len = 0
        for i in range(min(buf.capacity, len(test_sequences))):
            if i % (num_ranks - 1) == 0:
                buf.prepare_buffers(num_ranks)
            cur_len += len(test_sequences[i])
            for obs in test_sequences[i]:
                o, info = obs
                buf.append(o, {'rank': i % (num_ranks - 1) + 1})
        sampled = buf.sample_sequences(to_sample)
        self.assertEqual(len(sampled), to_sample)
        self.assertEqual(sampled[-1][0] is None, False)

    def test_sample_sequences_more(self):
        with open(r'C:\Users\burov\Projects\mas-project-burov-ay2122\test_seq_list.pkl', 'rb') as f:
            test_sequences = pickle.load(f)
        seq_capacity = 12
        num_ranks = 4
        env_memory_size = 8
        buf = EpisodeReplayBuffer(seq_capacity, num_ranks, env_memory_size)
        buf.prepare_buffers(num_ranks)
        cur_len = 0
        for i in range(min(buf.capacity, len(test_sequences))):
            if i % (num_ranks - 1) == 0:
                buf.prepare_buffers(num_ranks)
            cur_len += len(test_sequences[i])
            for obs in test_sequences[i]:
                o, info = obs
                buf.append(o, {'rank': i % (num_ranks - 1) + 1})
        sampled = buf.sample_sequences(buf.capacity + 1)
        self.assertEqual(len(sampled), buf.capacity)
        self.assertEqual(sampled[-1][0] is None, False)

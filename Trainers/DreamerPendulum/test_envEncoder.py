from unittest import TestCase
import tensorflow as tf
import numpy as np

from Trainers.DreamerPendulum.models import EnvEncoder, EnvDecoder


class TestEnvEncoder(TestCase):
    def test_forward_pass_decoder(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        decoder = EnvDecoder(50, 20)
        decoder.model.summary()
        embs = np.zeros((15, 4, 50))
        v = decoder(embs)
        self.assertEqual(v.shape, (15, 4, 3))

        embs = np.zeros((4, 50))
        v = decoder(embs)
        self.assertEqual(v.shape, (4, 3))

    def test_forward_pass_encoder(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        encoder = EnvEncoder(50, 20)
        encoder.model.summary()
        vf = np.zeros((15, 2, 3))
        res = encoder(vf)
        self.assertEqual(res.shape, (15, 2, 50))

        vf = np.zeros((2, 3))
        res = encoder(vf)
        self.assertEqual(res.shape, (2, 50))

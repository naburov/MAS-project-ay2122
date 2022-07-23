from unittest import TestCase
import tensorflow as tf
import numpy as np

from Trainers.Dreamer.models import EnvEncoder, EnvDecoder


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
        decoder = EnvDecoder(8, 50, 200, 32)
        decoder.model.summary()
        embs = np.zeros((15, 4, 50))
        vf, v = decoder(embs)
        self.assertEqual(v.shape, (15, 4, 952))

        embs = np.zeros((4, 50))
        vf, v = decoder(embs)
        self.assertEqual(v.shape, (4, 952))

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
        encoder = EnvEncoder(8, 50, 200, 32)
        encoder.model.summary()
        vf = np.zeros((15, 2, 11, 11, 16))
        v = np.zeros((15, 2, 952))
        res = encoder((vf, v))
        self.assertEqual(res.shape, (15, 2, 50))

        vf = np.zeros((2, 11, 11, 16))
        v = np.zeros((2, 952))
        res = encoder((vf, v))
        self.assertEqual(res.shape, (2, 50))

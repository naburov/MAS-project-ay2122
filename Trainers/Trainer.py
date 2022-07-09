from abc import ABC, abstractmethod


class TrainManager(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict_actions(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def on_epoch_end(self, epoch_n):
        pass

    @abstractmethod
    def append_observations(self, *data):
        pass

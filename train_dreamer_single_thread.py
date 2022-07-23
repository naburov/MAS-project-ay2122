from MyEnv import MyEnv
from Trainers.Dreamer.TrainManager import DreamerTrainManager
from config import *
from Trainers.DDPGTrainManager.logger import Logger

import tensorflow as tf

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

manager = DreamerTrainManager(checkpoint_dir=MODEL_CHECKPOINT_DIR,
                              buffer_path=BUFFER_CHECKPOINT_DIR,
                              memory_size=ENV_MEMORY_SIZE,
                              buffer_capacity=BUFFER_CAPACITY,
                              num_ranks=2,
                              batch_size=BATCH_SIZE)
logger = Logger()
print('Loaded: ', manager.buf.num_sequences, ' sequences')

env = MyEnv(visualize=False, memory_size=ENV_MEMORY_SIZE)
for epoch in range(EPOCHS):
    for num_episode in range(EPISODES_PER_EPOCH):
        is_random = False

        if manager.buf.num_sequences < BATCH_SIZE:
            is_random = True

        observation = env.reset()
        manager.on_episode_begin(epoch, num_episode)

        done = False
        r = 0
        while not done:
            actions = manager.predict_actions([observation])
            if is_random:
                actions = [env.action_space.sample()]
            old_observations = observation
            observation, reward, done, info = env.step(actions[0])
            r += reward
            manager.append_observations(
                (*old_observations, reward, *observation, actions[0]), {'rank': 1})

        print('Finished episode ', num_episode)
        logger.log2txt(REWARD_LOGS_PATH,
                       'Epoch: {0}, Ep n: {1}, Avg rew: {2}'.format(epoch, num_episode, r))

        if manager.buf.num_sequences > BATCH_SIZE:
            losses = manager.train_step()
            logger.logDict(LOSSES_LOGS_PATH, losses)

    manager.on_epoch_end(epoch)

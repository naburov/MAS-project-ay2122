from MyEnv import MyEnv
from Trainers.Dreamer.TrainManager import DreamerTrainManager
from config import *
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

manager = DreamerTrainManager(checkpoint_dir=MODEL_CHECKPOINT_DIR,
                              buffer_path=BUFFER_CHECKPOINT_DIR,
                              memory_size=ENV_MEMORY_SIZE,
                              buffer_capacity=BUFFER_CAPACITY,
                              num_ranks=2,
                              batch_size=BATCH_SIZE)

env = MyEnv(visualize=True, memory_size=ENV_MEMORY_SIZE, integrator_accuracy=INTEGRATOR_ACCURACY)
for num_episode in range(VALIDATE_EPISODES):

    observation = env.reset()

    done = False
    r = 0
    it_count = 0
    while not done:
        if it_count % N_REPEAT_ACIONS == 0:
            actions = manager.predict_actions([observation], training=False)
        it_count += 1

        observation, reward, done, info = env.step(actions[0])
        r += reward

    print('Finished episode {0}, reward is {1}'.format(num_episode, r))

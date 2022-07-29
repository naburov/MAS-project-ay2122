EPOCHS = 1000
EPISODES_PER_EPOCH = 10
ENV_MEMORY_SIZE = 1
TAU = 0.01
NOISE_STD_DEV = 0.2
BATCH_SIZE = 32
BUFFER_CAPACITY = 10000
TRAIN_STEPS = 100
SEQUENCE_LENGTH = 30
# BUFFER_CAPACITY = int(5e4)
MODEL_CHECKPOINT_DIR = r'C:\Users\burov\Projects\mas-project-burov-ay2122\checkpoints-dreamer'
BUFFER_CHECKPOINT_DIR = r'C:\Users\burov\Projects\mas-project-burov-ay2122\buf-2.pckl'
REWARD_LOGS_PATH = r'C:\Users\burov\Projects\mas-project-burov-ay2122\reward-dreamer.log'
LOSSES_LOGS_PATH = r'C:\Users\burov\Projects\mas-project-burov-ay2122\losses-dreamer.log'
# MODEL_CHECKPOINT_DIR = r'/mnt/mas-project-burov-ay2122/checkpoints'
# BUFFER_CHECKPOINT_DIR = r'/mnt/mas-project-burov-ay2122/buf.pckl'
# REWARD_LOGS_PATH = '/mnt/mas-project-burov-ay2122/rewards.txt'

N_REPEAT_ACIONS = 5

# net params
hidden = 256
determ = 48
stoch = 16
units = 400
num_actions = 22
kl_scale = 1.0
gamma = 0.99
lambda_ = 0.95
horizon = 15
env_memory_size = 1
embedding_size = 64
filters = 32
noise = 0.1

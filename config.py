EPOCHS = 1000
EPISODES_PER_EPOCH = 4
ENV_MEMORY_SIZE = 1
TAU = 0.01
MAX_STEPS = 1000
NOISE_STD_DEV = 0.2
BATCH_SIZE = 4
BUFFER_CAPACITY = 100000
TRAIN_STEPS = 100
SEQUENCE_LENGTH = 50
# BUFFER_CAPACITY = int(5e4)
# MODEL_CHECKPOINT_DIR = r'C:\Users\burov\Projects\mas-project-burov-ay2122\checkpoints-dreamer-2022-08-10'
MODEL_CHECKPOINT_DIR = r'D:\Загрузки'
# BUFFER_CHECKPOINT_DIR = r'D:\Загрузки\buf_saved.pckl'
BUFFER_CHECKPOINT_DIR = r'C:\Users\burov\Projects\mas-project-burov-ay2122\buf-sparse.pckl'
REWARD_LOGS_PATH = r'C:\Users\burov\Projects\mas-project-burov-ay2122\reward-dreamer.log'
LOSSES_LOGS_PATH = r'C:\Users\burov\Projects\mas-project-burov-ay2122\losses-dreamer.log'
DEBUG_LOGS_PATH = r'C:\Users\burov\Projects\mas-project-burov-ay2122\debug-dreamer.log'
H_DEBUG_LOGS_PATH = r'C:\Users\burov\Projects\mas-project-burov-ay2122\current-debug-dreamer.log'
PRETRAIN_STEPS = 0
DENSE_DECODER_NOISE = 1.0
VALUE_DENSE_DECODER_NOISE = 0.25
REWARD_DENSE_DECODER_NOISE = 0.1
ACTOR_NOISE = 0.1
DECODER_VF_STD = 0.5
DECODER_V_STD = 1.5

GRAD_CLIP_NORM = 100.0

ACTOR_LR = 1e-4
ENCODER_LR = 5e-5
DECODER_LR = 1e-5
VALUE_LR = 1e-4
DYNAMICS_LR = 2e-5
REWARD_LR = 1e-4
INTEGRATOR_ACCURACY = 1e-2
# MODEL_CHECKPOINT_DIR = r'/mnt/mas-project-burov-ay2122/checkpoints'
# BUFFER_CHECKPOINT_DIR = r'/mnt/mas-project-burov-ay2122/buf.pckl'
# REWARD_LOGS_PATH = '/mnt/mas-project-burov-ay2122/rewards.txt'

N_REPEAT_ACIONS = 5
BUF_SAVE_TIMEOUT = 3

# net params
hidden = 384
determ = 200
stoch = 30
units = 1000
num_actions = 22

kl_scale = 1.0
rec_scale = 1.0
rew_scale = 1.0

gamma = 0.99
lambda_ = 0.95
horizon = 10
env_memory_size = 1
embedding_size = 128
filters = 48
noise = 0.1

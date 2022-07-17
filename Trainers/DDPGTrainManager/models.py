import tensorflow as tf

from Trainers.simple_replay_buffer import VECTOR_OBS_LEN

NUM_ACTIONS = 22


def get_actor_model(env_memory_size):
    tgt_inputs = tf.keras.Input(shape=(11, 11, 2 * env_memory_size))
    vector_input = tf.keras.Input(shape=(VECTOR_OBS_LEN * env_memory_size,))

    conv_branch = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='linear')(tgt_inputs)
    conv_branch = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same')(conv_branch)
    conv_branch = tf.keras.layers.LeakyReLU()(conv_branch)
    conv_branch = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same')(conv_branch)
    conv_branch = tf.keras.layers.LeakyReLU()(conv_branch)
    conv_vector = tf.keras.layers.GlobalAveragePooling2D()(conv_branch)

    x = tf.keras.layers.Dense(1000, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4))(vector_input)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(1000, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(1000, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4))(x)
    concated = tf.keras.layers.Concatenate()([x, conv_vector])

    out = tf.keras.layers.Dense(NUM_ACTIONS, activation='sigmoid')(concated)
    return tf.keras.Model(inputs=[tgt_inputs, vector_input], outputs=out)


def get_critic_model(env_memory_size):
    tgt_inputs = tf.keras.Input(shape=(11, 11, 2 * env_memory_size))
    vector_input = tf.keras.Input(shape=(VECTOR_OBS_LEN * env_memory_size,))
    action_input = tf.keras.Input(shape=(NUM_ACTIONS,))

    conv_branch = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='selu')(tgt_inputs)
    conv_branch = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='selu')(conv_branch)
    conv_vector = tf.keras.layers.GlobalAveragePooling2D()(conv_branch)

    x = tf.keras.layers.Dense(1000, activation='selu')(vector_input)
    x = tf.keras.layers.Dense(1000, activation='selu')(x)
    x = tf.keras.layers.Dense(1000, activation='selu')(x)

    action_branch = tf.keras.layers.Dense(1000, activation='selu')(action_input)
    action_branch = tf.keras.layers.Dense(1000, activation='selu')(action_branch)
    action_branch = tf.keras.layers.Dense(1000, activation='selu')(action_branch)
    concated = tf.keras.layers.Concatenate()([x, conv_vector, action_branch])

    out = tf.keras.layers.Dense(1)(concated)
    return tf.keras.Model(inputs=[tgt_inputs, vector_input, action_input], outputs=out)

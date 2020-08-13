import tensorflow as tf



def build_model(dim):
    Input = tf.keras.layers.Input(shape=dim, dtype='float64')
    # Input = tf.keras.layers.Input(shape=dim, dtype='float64', sparse=True)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(Input)
    dropout = tf.keras.layers.Dropout(0.2)(dense1)
    dense2 = tf.keras.layers.Dense(1, activation='linear')(dropout)

    model = tf.keras.models.Model(inputs=Input, outputs=dense2)

    return model

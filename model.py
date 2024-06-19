import tensorflow as tf
from keras import layers


def build_model():
    my_model = tf.keras.Sequential([
        layers.Dense(50, activation=tf.nn.tanh, name='layer1'),
        layers.Dense(50, activation=tf.nn.tanh, name='layer2'),
        layers.Dense(10, activation=tf.nn.softmax, name='logits')
    ])
    return my_model

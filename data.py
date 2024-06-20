import os
import struct

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        # # for test
        # cv2.imshow('', images[0, :].reshape(28, 28))
        images = ((images / 255.) - .5) * 2
    return images, labels


# def norm_x_data(x_data):
#     mean_vals = np.mean(x_data, axis=0)
#     std_val = np.std(x_data)
#     x_centered = (x_data - mean_vals) / std_val
#     x_centered = ((x_centered / 255.) - .5) * 2
#     return x_centered


def create_batch_generator(x, y, batch_size=128, shuffle=False):
    x_copy = np.array(x)
    y_copy = np.array(y)

    if shuffle:
        data = np.column_stack((x_copy, y_copy))
        np.random.shuffle(data)
        x_copy = data[:, :-1]
        y_copy = data[:, -1].astype(int)

    # x_centered = norm_x_data(x_copy)

    y_one_hot = tf.one_hot(y_copy, 10)

    for i in range(0, x.shape[0], batch_size):
        yield x_copy[i:i + batch_size, :], y_one_hot[i:i + batch_size, :]


def display_mnist_images(images, labels, n):
    """Display the first n MNIST images with their labels"""
    num_columns = 10
    num_rows = (n + num_columns - 1) // num_columns
    plt.figure()
    plt.suptitle("MNIST samples")

    for i in range(n):
        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f'{labels[i]}')
        plt.axis('off')
        plt.subplots_adjust(top=0.88)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_path = "./dataset"
    train_img, train_label = load_mnist(data_path)
    display_mnist_images(train_img, train_label, 30)


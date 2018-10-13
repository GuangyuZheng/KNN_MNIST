import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def plot_image(image):
    img = image.reshape(28, 28)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'dataset')
    images, labels = load_mnist(path, 'train')
    print(images[0])
    print(labels[0])
    plot_image(images[0])
    plot_image(images[1])
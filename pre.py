import numpy as np
import cv2
import struct

def Load(kind):
    labelsSrc = './' + kind + '-labels.idx1-ubyte'
    imagesSrc = './' + kind + '-images.idx3-ubyte'
    with open(labelsSrc, 'rb') as lrc:
        magic, n = struct.unpack('>II', lrc.read(8))
        labels = np.fromfile(lrc, dtype=np.uint8)

    with open(imagesSrc, 'rb') as irc:
        magic, n, r, c = struct.unpack('>IIII', irc.read(16))
        images = np.fromfile(irc, dtype=np.uint8).reshape(len(labels), 28 * 28)


    images = np.array(images)
    images = images.astype(np.float32) / 255
    images = np.repeat(images, 3)
    images = np.reshape(images, (-1, 28, 28, 3))

    labels = np.reshape(np.array(labels), (-1, )).astype(np.int64)

    return images, labels


trainX, trainY = Load('train')
np.save("train_data.npy", trainX)
np.save("train_label.npy", trainY)

testX, testY = Load('t10k')
np.save("test_data.npy", testX)
np.save("test_label.npy", testY)

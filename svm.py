import numpy as np
import cv2
import struct
import matplotlib.pyplot as plt
import sklearn.svm as svm

def Load(kind):
    labelsSrc = './' + kind + '-labels.idx1-ubyte'
    imagesSrc = './' + kind + '-images.idx3-ubyte'
    with open(labelsSrc, 'rb') as lrc:
        magic, n = struct.unpack('>II', lrc.read(8))
        labels = np.fromfile(lrc, dtype=np.uint8)

    with open(imagesSrc, 'rb') as irc:
        magic, n, r, c = struct.unpack('>IIII', irc.read(16))
        images = np.fromfile(irc, dtype=np.uint8).reshape(len(labels), 28 * 28)

    return images, labels

    
    '''fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, )
    ax = ax.flatten()
    for i in range(10):
        img = images[labels == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()'''

def getFeature(images):
    from skimage.feature import hog
    X = []
    for img in images:
        img = img.reshape(28, 28)
        fd = hog(img, pixels_per_cell=(4, 4), 
                    cells_per_block=(3, 3), feature_vector=True)
        X.append(fd)

    X = np.array(X)
    return X

trainX, trainY = Load('train')
X = getFeature(trainX)
Y = np.reshape(np.array(trainY), (-1, ))
clf = svm.LinearSVC(C=1)
clf.fit(X, Y)

testX, testY = Load('t10k')
X = getFeature(testX)
Y = np.reshape(np.array(testY), (-1, ))
print('HOG: ', clf.score(X, Y))
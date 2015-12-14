from config import DATA_PATH
import lib.gen_input as gen_input
from image_processing import rgb_to_grayscale
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
import itertools as it
import tensorflow as tf
import cv2

COLOURS = it.cycle(('-bo', '-ro', '-go', '-ko', '-yo', '-co', '-mo',
                    '-bs', '-rs', '-gs', '-ks', '-ys', '-cs', '-ms',
                    '-b*', '-r*', '-g*', '-k*', '-y*', '-c*', '-m*'))

def define_plot(x_title='', y_title='', title='', label_prefix='', labels=''):
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.draw()
    plt.show()


def plot_accuracy(x, y, label=''):
    plt.ion()
    plt.plot(x, y, next(COLOURS), label=label)


def normalize(img):
    cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img

def color_channel_as_feature(x, y):
    '''dataset has two attributes:
          images - 2D multichannel
          labels - 1D target values
        Converts the images' color channels to extra feature dimensions
    '''
    # Size of Image * Num Colors
    num_dimensions = x.shape[1] * x.shape[2]
    x = np.reshape(x, newshape=(-1, num_dimensions))
    return x, y


def color_channel_as_example(x_img, y_img):
    '''
        x is images
        y is targets
        Converts the images' color channels to extra examples
    '''
    original_examples = x_img.shape[0]
    num_examples = x_img.shape[0] * x_img.shape[2]
    num_features = x_img.shape[1]
    x = np.zeros((num_examples, num_features))
    y = np.zeros((num_examples, 10))  # 10 Classes
    for i in range(x_img.shape[2]):
        x[i * original_examples: (i + 1) * original_examples, :] = x_img[:, :, i]
        y[i * original_examples:(i + 1) * original_examples, :] = y_img  # Duplicate targets num_colors times
    return x, y

def load_extras(sets):
    assert(type(sets) is list)
    extra = gen_input.read_data_sets(DATA_PATH + "extra_32x32_" + str(sets[0]) + ".mat", [1, 0, 0])
    for i in range(1, len(sets)):
        num = sets[i]
        dataset = gen_input.read_data_sets(DATA_PATH + "extra_32x32_" + str(num) + ".mat", [1, 0, 0])
        np.concatenate((extra.train.images, dataset.train.images), axis=0)
        np.concatenate((extra.train.labels, dataset.train.labels), axis=0)

    return extra

Ks = range(1, 62, 6)  # MIN=1 MAX=61
REDUCED_DIMENSIONS = [1,2]#,3,4,20]   # 1-4 determined by looking at PCA().fracs

# Optimals configs:
Ks = [7]
REDUCED_DIMENSIONS = [3]
def main():
    # div_fractions = [0.80, 0.0, 0.20]  # Fractions to divide data into the train, valid, and test sets
    train = gen_input.read_data_sets(DATA_PATH + "train_32x32.mat", [1, 0, 0])
    test = gen_input.read_data_sets(DATA_PATH + "test_32x32.mat", [0, 0, 1])
    #extra_train = load_extras(range(1,6))
    Xtr = train.train.images
    Ytr = train.train.labels
    #Xtr = np.concatenate((train.train.images, extra_train.train.images), axis=0)
    #Ytr = np.concatenate((train.train.labels, extra_train.train.labels), axis=0)
    Xte = test.test.images
    Yte = test.test.labels
    print "Loaded data!"

    # Convert to grayscale
    Xtr = rgb_to_grayscale(Xtr)
    Xte = rgb_to_grayscale(Xte)

    # Xtr = normalize(Xtr)
    # Xte = normalize(Xte)

    # Y attribute stores data projects into PCA space using all eigen vectors.
    # The eigen vectors are in decreasing order, so PCA().Y[:,0:x] returns the data projected to x-dimensional PCA space
    pca_train = PCA(Xtr)
    pca_test = PCA(Xte)
    print "eigenvector top weights ", pca_train.fracs[0:20]

    define_plot(x_title="K", y_title="Accuracy",
                title="Grayscale Test Accuracy vs K for Nearest Neighbors using PCA", label_prefix="K=")
    print "1b"
    plt.xlim((min(Ks) - 1, max(Ks) + 1))
    min_accuracy = 1.0
    max_accuracy = 0
    for reduced_dimension in REDUCED_DIMENSIONS:
        print "starting with PCA dim ", reduced_dimension
        Xtr = pca_train.Y[:, 0:reduced_dimension]
        Xte = pca_test.Y[:, 0:reduced_dimension]
        print "Done trimming to PCA dimension"

        num_dimensions = Xtr.shape[1]

        # Graph Input
        pl_x_train = tf.placeholder("float", shape=[None, num_dimensions])
        pl_x_test = tf.placeholder("float", shape=[num_dimensions])

        # Nearest Neighbor calculation using L1 Norm Distance
        distance = tf.reduce_sum(tf.abs(tf.add(pl_x_train, tf.neg(pl_x_test))), reduction_indices=1)

        neg_distance = tf.neg(distance)  # MIN(distance) = MAX(neg_distance)

        # Couldn't get this to work: Wanted to use top_k then use scipy's mode method in loop below
        # largest_neg_distance, top_classes = tf.nn.top_k(neg_distances, K)
        # Predict: Get index of the most frequent class (Nearest Neighbor)
        # prediction = tf.argmin(distance, 0)

        print "Init Session..."
        # Get session ready
        init = tf.initialize_all_variables()
        session = tf.Session()
        session.run(init)

        print "Starting training/testing ", len(Xte), " examples"
        accuracies = []
        used_ks = []
        for K in Ks:
            print "Starting K = ", K
            num_correct = 0
            # loop over test data
            for i in range(len(Xte)):
                # Get nearest neighbor
                # nn_index = session.run(prediction, feed_dict={pl_x_train: Xtr, pl_x_test: Xte[i, :]})
                neg_distances = session.run(neg_distance, feed_dict={pl_x_train: Xtr, pl_x_test: Xte[i, :]})
                top_classes_index = np.argpartition(neg_distances, -K)[-K:]

                top_class_index, count = mode(top_classes_index)
                top_class_index = top_class_index[0]  # Unbox from array

                if (i % 1000 == 0):
                    print "Test", (i + 1), "Prediction:", np.argmax(Ytr[top_class_index]), "True Class:", np.argmax(Yte[i])

                # Get nearest neighbor class label and compare it to its true label
                if np.argmax(Ytr[top_class_index]) == np.argmax(Yte[i]):
                    num_correct += 1

            accuracy = float(num_correct) / len(Xte)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
            elif accuracy < min_accuracy:
                min_accuracy = accuracy

            print "Accuracy:", accuracy
            accuracies.append(accuracy)
            used_ks.append(K)
            plot_accuracy(used_ks, accuracies)

        plot_accuracy(Ks, accuracies, label="PCA " + str(reduced_dimension))
        print Ks
        print accuracies
        plt.ylim((min_accuracy - 0.1 * min_accuracy, max_accuracy + 0.1 * max_accuracy))
        plt.legend(loc='upper right')
        plt.ioff();
        raw_input('Press Enter to exit.')
        plt.savefig('./plots/knn/grayscale1.png', bbox_inches='tight')


if __name__ == '__main__':
    main()

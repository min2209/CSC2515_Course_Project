import numpy as np
from scipy.stats import mode
import tensorflow as tf
import lib.gen_input as gen_input


def color_channel_as_feature(dataset):
    '''dataset has two attributes:
          images - 2D multichannel
          labels - 1D target values
        Converts the images' color channels to extra feature dimensions
    '''
    # Size of Image * Num Colors
    x = dataset.images
    y = dataset.labels
    num_dimensions = x.shape[1] * x.shape[2]
    x = np.reshape(x, newshape=(-1, num_dimensions))
    return x, y

def color_channel_as_example(dataset):
    '''dataset has two attributes:
          images - 2D multichannel
          labels - 1D target values
        Converts the images' color channels to extra examples
    '''
    x_img = dataset.images
    y_img = dataset.labels
    original_examples = x_img.shape[0]
    num_examples = x_img.shape[0] * x_img.shape[2]
    num_features = x_img.shape[1]
    x = np.zeros((num_examples, num_features))
    y = np.zeros((num_examples, 10)) # 10 Classes
    for i in range(x_img.shape[2]):
        x[i*original_examples : (i+1)*original_examples, :] = x_img[:,:,i]
        y[i*original_examples:(i+1)*original_examples, :] = y_img   # Duplicate targets num_colors times
    return x, y

K = 5
div_fractions = [0.80, 0.0, 0.2]  # Fractions to divide data into the train, valid, and test sets
inputs = gen_input.read_data_sets("/media/min/Data/SVHN/Format2/train_32x32.mat", div_fractions)

print "Loaded data!"

# Reshape images to 1D
Xtr, Ytr = color_channel_as_feature(inputs.train)
Xte, Yte = color_channel_as_feature(inputs.test)

print "Done converting input to appropriate shapes"
num_dimensions = Xtr.shape[1]
# Graph Input
pl_x_train = tf.placeholder("float", shape=[None, num_dimensions])
pl_x_test = tf.placeholder("float", shape=[num_dimensions])

# Nearest Neighbor calculation using L1 Norm Distance
distance = tf.reduce_sum(tf.abs(tf.add(pl_x_train, tf.neg(pl_x_test))), reduction_indices=1)

neg_distance = tf.neg(distance)     # MIN(distance) = MAX(neg_distance)

# Couldn't get this to work: Wanted to use top_k then use scipy's mode method in loop below
#largest_neg_distance, top_classes = tf.nn.top_k(neg_distances, K)
# Predict: Get index of the most frequent class (Nearest Neighbor)
#prediction = tf.argmin(distance, 0)

num_correct = 0

print "Init Session..."
# Get session ready
init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

print "Starting training/testing"
# loop over test data
for i in range(len(Xte)):
    # Get nearest neighbor
    #nn_index = session.run(prediction, feed_dict={pl_x_train: Xtr, pl_x_test: Xte[i, :]})
    neg_distances = session.run(neg_distance, feed_dict={pl_x_train: Xtr, pl_x_test: Xte[i, :]})
    top_classes_index = np.argpartition(neg_distances, -K)[-K:]

    top_class_index, count = mode(top_classes_index)
    top_class_index = top_class_index[0]    # Unbox from array
    # Get nearest neighbor class label and compare it to its true label
    print "Test", (i + 1), "Prediction:", np.argmax(Ytr[top_class_index]), "True Class:", np.argmax(Yte[i])
    # Calculate accuracy
    if np.argmax(Ytr[top_class_index]) == np.argmax(Yte[i]):
        num_correct += 1
print "Done!"
print "Accuracy:", (float(num_correct) / len(Xte))

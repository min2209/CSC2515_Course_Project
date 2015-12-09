from config import DATA_PATH
import lib.gen_input as gen_input

import numpy as np
from math import ceil
from scipy.stats import mode
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA

import tensorflow as tf
import cv2

# COLOURS = ['b-', 'r-', 'g-', 'k-', 'y-', 'c-', 'm-']
def plot(xs, ys, x_title='', y_title='', title='', label_prefix='', labels=''):
  plt.ion()
  plt.figure(1)
  plt.clf()
  plt.axis((0, ceil(max(xs) + 0.1 * max(xs)), 0, ceil(np.max(ys) + 0.1 * np.max(ys))))
  plt.plot(xs, ys, '-o', label='Test Set')
  plt.xlabel(x_title)
  plt.ylabel(y_title)
  plt.title(title)
  plt.draw()
  plt.show()

def plot_accuracy(x, y):
    plot(x, y,
         x_title="K", y_title = "Accuracy",
         title="Test Accuracy vs K for Nearest Neighbors", label_prefix="K=", labels=x)

def normalize(img):
    cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img

def rgb_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

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
    y = np.zeros((num_examples, 10)) # 10 Classes
    for i in range(x_img.shape[2]):
        x[i*original_examples : (i+1)*original_examples, :] = x_img[:,:,i]
        y[i*original_examples:(i+1)*original_examples, :] = y_img   # Duplicate targets num_colors times
    return x, y

Ks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

def main():
  # div_fractions = [0.80, 0.0, 0.20]  # Fractions to divide data into the train, valid, and test sets
  train = gen_input.read_data_sets(DATA_PATH + "train_32x32.mat", [1.0,0,0])
  test = gen_input.read_data_sets(DATA_PATH + "test_32x32.mat", [0,0,1.0])
  Xtr = train.train.images
  Ytr = train.train.labels
  Xte = test.test.images
  Yte = test.test.labels
  print "Loaded data!"

  print Xtr.shape, " ", Xte.shape
  # Convert to grayscale
  Xtr = rgb_to_grayscale(Xtr)
  Xte = rgb_to_grayscale(Xte)
  print Xtr.shape, " ", Xte.shape
  Xtr = PCA(Xtr).Y
  Xte = PCA(Xte).Y
  print Xtr.shape, " ", Xte.shape

  return
# Reshape images to 1D
  Xtr, Ytr = color_channel_as_example(Xtr, Ytr)
  Xte, Yte = color_channel_as_example(Xte, Yte)

#Xtr = normalize(Xtr)
#Xte = normalize(Xte)

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
          #nn_index = session.run(prediction, feed_dict={pl_x_train: Xtr, pl_x_test: Xte[i, :]})
          neg_distances = session.run(neg_distance, feed_dict={pl_x_train: Xtr, pl_x_test: Xte[i, :]})
          top_classes_index = np.argpartition(neg_distances, -K)[-K:]

          top_class_index, count = mode(top_classes_index)
          top_class_index = top_class_index[0]    # Unbox from array

          if (i % 10 == 0):
              print "Test", (i + 1), "Prediction:", np.argmax(Ytr[top_class_index]), "True Class:", np.argmax(Yte[i])

          # Get nearest neighbor class label and compare it to its true label
          if np.argmax(Ytr[top_class_index]) == np.argmax(Yte[i]):
              num_correct += 1

      accuracy = float(num_correct) / len(Xte)
      print "Accuracy:", accuracy
      accuracies.append(accuracy)
      used_ks.append(K)
      plot_accuracy(used_ks, accuracies)

  plot_accuracy(Ks, accuracies)
  raw_input('Press Enter to exit.')

if __name__ == '__main__':
  main()

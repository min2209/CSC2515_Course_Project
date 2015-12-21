"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os
import numpy
from .load_mat import load_mat
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class DataSet(object):
  def __init__(self, images, labels, normalize=True):
    assert images.shape[0] == labels.shape[0], (
        "images.shape: %s labels.shape: %s" % (images.shape,
                                               labels.shape))
    self._num_examples = images.shape[0]
    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    assert images.shape[3] == 3
    images = images.reshape(images.shape[0],
                            images.shape[1] * images.shape[2], images.shape[3])
    if normalize:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)
    self._images = numpy.array(images)
    self._labels = numpy.array(labels)
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    to_return = self._images[start:end], self._labels[start:end]
    return to_return

def reflect(a):
  '''Used if, when loading data, you do not want a one-hot encoding (as this was hard-coded first so to keep backwards compatibility)'''
  return a

def read_data_sets(path_to_mat, div_fractions, normalize = True, label_encoding = dense_to_one_hot):
  class DataSets(object):
    pass
  assert sum(div_fractions) <= 1  # Allow using a fraction of data set
  data_sets = DataSets()

  # load all input values. roll image index to position 0 in ndarray
  input_dict = load_mat(path_to_mat)
  input_all = numpy.rollaxis(input_dict["X"], 3)

  # load all target values. change target values to onehot
  target_all = label_encoding(input_dict["y"])

  train_end_index = int(div_fractions[0]*input_all.shape[0])
  valid_end_index = int((div_fractions[0] + div_fractions[1])*input_all.shape[0])
  test_end_index = int(sum(div_fractions) * input_all.shape[0])

  data_sets.train = DataSet(input_all[0:train_end_index], target_all[0:train_end_index], normalize=normalize)
  data_sets.validation = DataSet(input_all[train_end_index:valid_end_index], target_all[train_end_index:valid_end_index], normalize=normalize)
  data_sets.test = DataSet(input_all[valid_end_index:test_end_index], target_all[valid_end_index:test_end_index], normalize=normalize)
  return data_sets



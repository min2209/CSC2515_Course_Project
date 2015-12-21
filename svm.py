from config import DATA_PATH, IMAGE_DIMENSION
import lib.gen_input as gen_input
import image_processing as feature_extraction
from bag_of_features import FeatureBag
import numpy as np
import cv2
import scipy.io as scio
import lib.mnist as input_data
import matplotlib.pyplot as plt

class SVM:
    '''Wrapper for OpenCV SVM'''
    def __init__(self, params = dict(kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC,C = 1)):
        self._model = cv2.SVM()
        self._params = params

    def train(self, data, responses):
        self._model.train(np.float32(data), np.float32(responses), params = self._params)

    def predict(self, samples):
        return np.float32([self._model.predict(np.float32(s)) for s in samples])


NUM_CLUSTERS = range(5, 46, 5)
NUM_CLUSTERS = [40]
def main():
    trim = 4
    train = gen_input.read_data_sets(DATA_PATH + "train_32x32.mat", [1, 0, 0], False, gen_input.reflect)
    test = gen_input.read_data_sets(DATA_PATH + "test_32x32.mat", [0, 0, 1], False, gen_input.reflect)
    #extra_train = load_extras(range(1,6))
    Xtr = train.train.images.reshape(-1, IMAGE_DIMENSION, IMAGE_DIMENSION, 3)
    Ytr = train.train.labels

    #Xtr = np.concatenate((Xtr, extra_train.train.images), axis=0)
    #Ytr = np.concatenate((Ytr, extra_train.train.labels), axis=0)
    Xte = test.test.images.reshape(-1, IMAGE_DIMENSION, IMAGE_DIMENSION, 3)
    Yte = test.test.labels
    '''
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    Xtr = mnist.train.images
    Ytr = mnist.train.labels
    Xte = mnist.test.images
    Yte = mnist.test.labels
    '''
    # Ytr_2 = np.where(Ytr == 0)[0]
    # Ytr_3 = np.where(Ytr == 7)[0]
    # indices = np.concatenate((Ytr_2, Ytr_3))
    # Ytr = Ytr[indices]
    # Xtr = Xtr[indices,:]#,:,:]
    # Yte_2 = np.where(Yte == 0)[0]
    # Yte_3 = np.where(Yte == 7)[0]
    # indices = np.concatenate((Yte_2, Yte_3))
    # Yte = Yte[indices]
    # Xte = Xte[indices,:]#,:,:]

    print "Done loading data ", Xtr.shape[0], Xte.shape[0]


    # Convert images to grayscale
    t1 = feature_extraction.rgb_to_grayscale(Xtr)
    t2 = feature_extraction.rgb_to_grayscale(Xte)
    print "Done RGB"

    for trim in [4,6,8,10,12,14]:
        Xtr = t1[:,:,trim:-trim]
        Xte = t2[:,:,trim:-trim]

        '''
        # Extract features
        Xtr_feature, Xtr_feature_per_example, Xtr_duds = feature_extraction.all_extract_surf(Xtr)
        Xte_feature, Xte_feature_per_example, Xte_duds = feature_extraction.all_extract_surf(Xte)
        print "Done Feature Extraction ", Xtr_duds.shape, " ", Xte_duds.shape


        '''
        '''
        Xtr_feature = np.delete(Xtr_feature, Xtr_duds, axis=0)
        Xtr_feature_per_example = np.delete(Xtr_feature_per_example, Xtr_duds, axis=0)
        Xte_feature = np.delete(Xte_feature, Xte_duds, axis=0)
        Xte_feature_per_example = np.delete(Xte_feature_per_example, Xte_duds, axis=0)
        Ytr = np.delete(Ytr, Xtr_duds, axis=0)
        Yte = np.delete(Yte, Xte_duds, axis=0)

        '''

        # Extract HoG features
        Xtr_feature, Xtr_feature_per_example = feature_extraction.all_extract_hog(Xtr)
        Xte_feature, Xte_feature_per_example = feature_extraction.all_extract_hog(Xte)
        print "Done Feature Extraction "


        accuracies = {}
        for CLUSTER_COUNT in NUM_CLUSTERS:
            print "Starting config ", CLUSTER_COUNT
            fb = FeatureBag(CLUSTER_COUNT)
            #fb.set_data(Xtr_feature)
            print "Begin Fitting"
            fb.fit(Xtr_feature)
            print "Done Fitting"
            train_features = np.array(fb.predict_feature_vectors(Xtr_feature_per_example[0]))
            for i in range(1, len(Xtr_feature_per_example)):
                feature = Xtr_feature_per_example[i]
                cluster_desc = fb.predict_feature_vectors(feature)
                train_features = np.vstack((train_features, cluster_desc))

            test_features = np.array(fb.predict_feature_vectors(Xte_feature_per_example[0]))
            for i in range(1, len(Xte_feature_per_example)):
                feature = Xte_feature_per_example[i]
                cluster_desc = fb.predict_feature_vectors(feature)
                test_features = np.vstack((test_features, cluster_desc))

            print "Starting SVM"
            svm = SVM()
            svm.train(train_features, Ytr)
            predictions = svm.predict(test_features).reshape(Yte.shape)

            print "Done SVM predictions"

            # Assume Yte is dense encoding, comparing elements not arrays for equality
            accuracy = float(np.sum(predictions == Yte)) * 100.0 / Yte.shape[0]

            print "trim ", trim
            print "Accuracy = ", accuracy
            accuracies[CLUSTER_COUNT] = accuracy

        print accuracies

if __name__ == "__main__":
    main()

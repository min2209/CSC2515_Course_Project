from config import DATA_PATH, IMAGE_DIMENSION
import lib.gen_input as gen_input
import image_processing as feature_extraction
import plot_helper as plotter
from bag_of_features import FeatureBag
import numpy as np
import cv2

class SVM:
    '''Wrapper for OpenCV SVM'''
    def __init__(self, params = dict(kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC,C = 1)):
        self._model = cv2.SVM()
        self._params = params

    def train(self, data, responses):
        self.model.train(data, responses, params = self._params)

    def predict(self, samples):
        return np.float32([self.model.predict(s) for s in samples])


NUM_CLUSTERS = range(20, 100, 20)
NUM_ITERATIONS = 500
def main():
    train = gen_input.read_data_sets(DATA_PATH + "train_32x32.mat", [1, 0, 0], gen_input.reflect)
    test = gen_input.read_data_sets(DATA_PATH + "test_32x32.mat", [0, 0, 1], gen_input.reflect)
    #extra_train = load_extras(range(1,6))
    Xtr = train.train.images.reshape(-1, IMAGE_DIMENSION, IMAGE_DIMENSION, 3)
    Ytr = train.train.labels
    #Xtr = np.concatenate((Xtr, extra_train.train.images), axis=0)
    #Ytr = np.concatenate((Ytr, extra_train.train.labels), axis=0)
    Xte = test.test.images.reshape(-1, IMAGE_DIMENSION, IMAGE_DIMENSION, 3)
    Yte = test.test.labels

    # Convert images to grayscale
    Xtr = feature_extraction.rgb_to_grayscale(Xtr)
    Xte = feature_extraction.rgb_to_grayscale(Xte)

    # Extract SIFT features
    Xtr_sift = []
    for img in Xtr:
        Xtr_sift.append(feature_extraction.extract_sift(img))

    Xte_sift = []
    for img in Xte:
        Xte_sift.append(feature_extraction.extract_sift(img))

    Xtr_sift = np.array(Xtr_sift)
    Xte_sift = np.array(Xte_sift)

    sift_fb = FeatureBag(NUM_CLUSTERS, NUM_ITERATIONS)
    sift_fb.set_data(Xtr_sift)
    sift_fb.fit()
    train_feature = sift_fb.predict_feature_vectors(Xtr_sift)
    test_feature = sift_fb.predict_feature_vectors(Xte_sift)

    clf = SVM()
    clf.train(train_feature, Ytr)
    predictions = clf.predict(test_feature)

    #plotter.define_plot(x_title="K", y_title="Accuracy",
    #            title="Accuracy for SVM using SIFT Features", label_prefix="")

    # Assume Yte is dense encoding, comparing elements not arrays for equality
    accuracy = float(np.sum(predictions == Yte)) / Yte.shape[0]

    print "Accuracy = ", accuracy

if __name__ == "__main__":
    main()
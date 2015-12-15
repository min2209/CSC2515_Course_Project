from scipy.cluster.vq import kmeans2
import numpy as np

class FeatureBag:

    def __init__(self, num_clusters, max_iterations):
        self._num_clusters = num_clusters
        self._max_iterations = max_iterations
        self._data = np.array([])

    def add_data(self, data):
        self._data = np.concatenate((self._data, data), axis=0)

    def set_data(self, data):
        self._data = np.array(data)

    def fit(self):
        self._centroids, self._data_centroid_labels = kmeans2(self._data, self._num_clusters, self._max_iterations)

    def predict_feature_vectors(self, features):
        '''
        compute the feature vector for one example with the specified features
        distances to centroids are computed using euclidean distance metric
        :param features: NUM_FEATURES X NUM_DIMENSIONS numpy array
        :return: 1 x NUM_CLUSTERS (binary) numpy array representing feature vector
        '''

        NUM_FEATURES = features.shape[0]
        NUM_CLUSTERS, NUM_DIMENSIONS = self._centroids.shape

        # centroids is NUM_CLUSTERS x NUM_DIMENSION x NUM_FEATURES
        centroids = np.repeat(self._centroids[:,:,np.newaxis], NUM_FEATURES, axis=2)

        # features has same dimension as centroids
        features = np.repeat(features.reshape(1, NUM_DIMENSIONS, -1), NUM_CLUSTERS, axis=0)

        # compute distances over dimensions (so there is a distance for each (cluster, feature) combination
        #  should be NUM_CLUSTERS x NUM_FEATURES
        distance = np.sum((centroids - features) ** 2, axis=1)

        # 1 x NUM_FEATURES representing closest cluster
        closest_clusters = np.argmin(distance, axis=0)  # Take minimum distance index over all features

        # 1 x NUM_CLUSTERS
        feature_vector = np.zeros((1, NUM_CLUSTERS))

        # Can do either counts, or a binary representation
        for cluster in closest_clusters:
            feature_vector[cluster] = 1

        return feature_vector

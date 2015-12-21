from scipy.cluster.vq import kmeans
import numpy as np

class FeatureBag:

    def __init__(self, num_clusters):
        self._num_clusters = num_clusters

    def fit(self, data):
        self._centroids, self._data_centroid_labels = kmeans(data, self._num_clusters)

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
            feature_vector[0, cluster] += 1

        return feature_vector

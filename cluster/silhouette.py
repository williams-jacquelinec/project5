import numpy as np
from scipy.spatial.distance import cdist
# from .kmeans import KMeans

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """
        self.metric = metric

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        self.X = X 
        self.y = y 

        fit_model_centroids = [[ 7.57041575 -2.9889736 ], [ 8.90376386  3.92500241], [ 6.51760379 -4.3566827 ], [-5.29167043 -0.5010371 ]]

        clusters_ = [[] for _ in range(num_clusters)]

        a_list = []
        b_list = []
        max_list = []
        silhouette_score = []
        
        # initialize what point belongs in what cluster
        for x in range(self.X.shape[0]):
            clusters_[self.y[x]].append(self.X[x])

        
        # silhouette score = (b(j) - a(j))/max{a(j), b(j)}
        # step 1: calculate a(j) 
        for i in range(self.X.shape[0]):
            a_distance = 0
            b_distance = 0

            cluster_label = self.y[i]

            # calculating a(j) --> average distance btwn j (point) and all other points in its own cluster
            point_to_point_dist = cdist([self.X[i]], clusters_[cluster_label], self.metric)

            for j in point_to_point_dist[0]:
                a_distance += j

            average_dist_a = a_distance/len(clusters_[cluster_label])
            a_list.append(average_dist_a)

            # calculating b(j) --> average distance from point (j) to all points in the next closest cluster
            # figuring out what the closest cluster is (based on distance to next closest centroid)
            dist_to_cent = cdist([self.X[i]], fit_model_centroids, self.metric)
            sorted_dist_to_cent = np.sort(dist_to_cent)

            min_distance = sorted_dist_to_cent[0][1]
            min_distance_idx = dist_to_cent[0].index(min_distance)

            # calculating mean distance to points in closest cluster
            dist_to_closest_cluster = cdist([self.X[i]], clusters_[min_distance_idx], self.metric)

            for k in dist_to_closest_cluster[0]:
                b_distance += k 

            average_dist_b = b_distance/len(clusters_[min_distance_idx])
            b_list.append(average_dist_b)

            # calculating max value between a(j) and b(j)
            max_value = max(a_list[i], b_list[i])
            max_list.append(max_value)

            # calculating silhouette score
            score = (b_list[i] - a_list[i]) / max_list[i]
            silhouette_score.append(score)

        return silhouette_score



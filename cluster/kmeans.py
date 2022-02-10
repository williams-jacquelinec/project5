import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.k = k 
        self.metric = metric 
        self.tol = tol 
        self.max_iter = max_iter 


    def fit(self, mat: np.ndarray):
        """
        This method fits the kmeans algorithm onto a provided 2D matrix.

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        self.mat = mat
        self.centroid = []

        # list to store mean squared error for each iteration
        mse_list = []

        # initialize centroids (random indices) 
        random_indices = np.random.choice(self.mat.shape[0], self.k, replace=False)

        # creating a 2-D matrix of the randomly selected centroids
        for i in random_indices:
            self.centroid.append(self.mat[i])

        # initializing count object for while loop (while number of iterations has not reached max_iters)
        count = 0

        while count < self.max_iter:
            
            # creating empty clusters
            # list of sample indices for each cluster
            self.clusters = [[] for _ in range(self.k)]

            # creating a matrix of distances (observations to centroids)
            distance_matrix = cdist(self.mat, self.centroid, self.metric)

            #assigning minimum distances to respective clusters
            for i in range(distance_matrix.shape[0]):

                # calculating the index of the minimum value
                min_index = np.argmin(distance_matrix[i])
                
                # adding original observation to respective cluster
                self.clusters[min_index].append(self.mat[i])

            # WAY TO BREAK WHILE LOOP:
            # calculating the MSE!
            mean_squared_error = self._private_get_error()
            mse_list.append(mean_squared_error)

            if len(mse_list) > 1:
                difference = mean_squared_error - mse_list[-2]
                if abs(difference) < self.tol:
                    count += self.max_iter
                else:
                    count += 1
            elif len(mse_list) == 1:
                count += 1

            # figure out the new centroids
            centroids_old = self.centroid
            self.centroid = self._private_get_centroids(self.clusters)




    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        This method predicts the cluster labels for a provided 2D matrix.

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        self.mat = mat 
        cluster_labels = []

        # calculating distance between matrix observations and centroids (from the fit model)
        distance_matrix = cdist(self.mat, self.centroid, self.metric)

        for i in range(self.mat.shape[0]):
            min_index = np.argmin(distance_matrix[i])

            cluster_labels.append(min_index)

        return cluster_labels


        
    def _private_get_error(self) -> float:
        """
        returns the final mean-squared error of the fit model

        outputs:
            float
                the mean-squared error of the fit model
        """
        overall_dist = 0
        mse = 0

        for i in range(self.k):
            point_to_cent = cdist([self.centroid[i]], self.clusters[i], self.metric)

            for j in point_to_cent[0]:
                overall_dist += j

            mse += overall_dist/len(self.clusters[i])

        return mse 

    def _private_get_centroids(self, clusters) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        centroids = np.zeros((self.k, self.mat.shape[1]))
   
        for idx, cluster in enumerate(self.clusters):  
            cluster_mean = np.mean(cluster, axis=0)
            centroids[idx] = cluster_mean

        return centroids 






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

            # figure out the new centroids
            centroids_old = self.centroid
            self.centroid = self._new_centroids(self.clusters)

            # WAY TO BREAK WHILE LOOP:
            # calculating the MSE!
            mean_squared_error = self._get_error(centroids_old, self.centroid)
            print(mean_squared_error)
            mse_list.append(mean_squared_error)

            if len(mse_list) > 1:
                difference = mean_squared_error - mse_list[-2]
                if abs(difference) < self.tol:
                    count += self.max_iter
                else:
                    count += 1
            elif len(mse_list) == 1:
                count += 1



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


    def _new_centroids(self, clusters):
        """
        This function returns the new centroids of k clusters.

        input:
            clusters: np.ndarray
                2-D matrix of clusters
        output:
            centroids: np.ndarray
                2-D matrix of new centroids
        """
        centroids = np.zeros((self.k, self.mat.shape[1]))
   
        for idx, cluster in enumerate(self.clusters):  
            cluster_mean = np.mean(cluster, axis=0)
            centroids[idx] = cluster_mean

        return centroids  

        
    def _get_error(self, mat_original, mat_calculated) -> float:
        """
        returns the final mean-squared error of the fit model

        outputs:
            float
                the mean-squared error of the fit model
        """

        # will error be calculated for each cluster or averaged over the k clusters after each iteration?
        # do you only calculate this at the end of all iterations? and then call this function in the predict() function?
        # is this used on the centroids (since they change after every iteration, more or less)

        # calculate mean squared error between old centroids and newly assigned centroids?

        # self.mat_original = centroids_old
        # self.mat_calculated = self.centroid

        self.mat_original = mat_original
        self.mat_calculated = mat_calculated

        MSE = np.square(np.subtract(self.mat_original, self.mat_calculated)).mean()

        return MSE

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroid



# from utils import make_clusters
# from silhouette import score

# clusters, labels = make_clusters(k=4, scale=1)
# print(clusters[0])

# random_distance = cdist([clusters[0]], clusters[0:3], 'euclidean')
# print(random_distance)
# print(np.sort(random_distance))
# km = KMeans(k=4)
# km_fit = km.fit(clusters)
# print((km_fit))

# assert cluster != km_fit

# pred = km.predict(clusters)
# print(pred)
# print(pred)

# sil_score = km.score(clusters, pred)
# print(sil_score)












        # print("og distance matrix")
        # print(distance_matrix)

        # #assigning minimum distances to respective clusters
        # for i in range(distance_matrix.shape[0]):

        #     # calculating the index of the minimum value
        #     min_index = np.argmin(distance_matrix[i])
            
        #     # adding original observation to respective cluster
        #     self.clusters[min_index].append(self.mat[i])




        # return self.clusters[0]
        # print("sorted distance matrix")
        # print(np.sort(distance_matrix))
        # return self.centroid
        # print(cluster_labels)


            # count += 1

            # centroids_dist = cdist(centroids_old, self.centroid, self.metric)
            # # print(centroids_old)
            # # print(self.centroid)
            # # print(centroids_dist)
            # _cent_dist = 0
            # for i in range(len(centroids_dist)):
            #     _cent_dist += centroids_dist[i][i]
            #     # print(centroids_dist[i][i])

            # # print(_cent_dist)

            # if _cent_dist == 0:
            #     count += self.max_iter
            # else:
            #     count += 1

            # print(count)

        # after the while loop is complete, calculate the mean-squared errors

        # finally, return the self.centroid (but maybe you don't need to do this)
        # return self.centroid





# Write your k-means unit tests here

# Importing Dependencies
import pytest
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)
import numpy as np

def test_kmeans():
    """
    This method is to test the functionality of the kmeans algorithm
    """
    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)

    # checking that there is a label for each row in clusters
    labels = len(pred)
    observations = clusters.shape[0]

    assert labels == observations 

    # checking that the number of unique labels is equal to the number of clusters
    assert len(set(pred)) == 4
    
    

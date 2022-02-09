# write your silhouette score unit tests here

# Importing Dependencies
import pytest
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)
import numpy as np

def test_silhouette():
    """
    This method is to test the functionality of the Silhouette().score() method. 
    """

    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)
    
    # checking the value of the silhouette scores
    for i in scores:
        assert -1 <= i <= 1

    # making sure there is a silhouette score for each row of the cluster matrix
    observations = clusters.shape[0]
    _scores_ = len(scores)

    assert observations == scores

    # plotting results
    # plot_multipanel(clusters, labels, pred, scores)

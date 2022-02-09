"""
BMI203: Biocomputing algorithms Winter 2022
Assignment 5: Implementation of KMeans and Silhouette Scoring
"""

from .kmeans import KMeans
from .silhouette import Silhouette
from .utils import (
        make_clusters, 
        plot_clusters,
        plot_multipanel)

__version__ = '0.1.0'
import math
import numpy as np
import pandas as pd

DATASET_PATH = "_datasets/Mall_Customers.csv"
NUM_CLUSTERS = 3
NUM_ITERATIONS = 100
NUM_DIMENSIONS = 4
INFINITE = 10000000000


class K_Means:
    def __init__(self, num_clusters, dataset, num_iterations, ignore_columns):
        self.centroids = []
        self.num_clusters = num_clusters
        self.num_iterations = num_iterations
        self.num_dimensions = len(dataset.columns)
        self.dataset = self.format_dataset(ignore_columns, dataset)

    def initialize_centroids(self):
        inital_centroids = []
        i = 0

        while i < self.num_clusters:
            new_centroid = np.random.rand(self.num_dimensions)
            inital_centroids.append(new_centroid)
            i += 1

        self.centroids = inital_centroids

    def nearest_centroid(self, x1):
        min_distance = INFINITE
        all_centroids = self.centroids
        nearest_centroid = all_centroids[0]

        for centroid in all_centroids:
            current_distance = self.euclidian_distance(centroid, x1)
            if(current_distance < min_distance):
                min_distance = current_distance
                nearest_centroid = centroid

        return nearest_centroid

    def euclidian_distance(x1, x2):
        if len(x1) == 0 or len(x2) == 0:
            raise Exception("Both vectors must have at least 01 dimention")

        if len(x1) != len(x2):
            raise Exception("Vectors have different dimensions")

        somatory = 0
        dimension = 0
        num_all_dimensions = len(x1)

        while dimension < num_all_dimensions:
            somatory += math.pow((x1[dimension] - x2[dimension]), 2)
            dimension += 1

        return math.sqrt(somatory)
    
    def format_dataset(self, ignore_columns, dataset):
        if(not ignore_columns):
            return dataset
        
        return dataset.drop(columns=ignore_columns)
        


dataset = pd.read_csv(DATASET_PATH)
kmeans = K_Means(NUM_CLUSTERS, dataset, NUM_ITERATIONS, ignore_columns=['CustomerID'])
kmeans.initialize_centroids()

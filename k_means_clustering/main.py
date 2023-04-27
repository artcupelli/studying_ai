import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

DATASET_PATH = "_datasets/Mall_Customers.csv"
NUM_CLUSTERS = 3
NUM_ITERATIONS = 100
NUM_DIMENSIONS = 4
INFINITE = 10000000000
INVALID_CENTROID = -1


class K_Means:
    def __init__(self, num_clusters, dataset, num_iterations):
        self.centroids = []
        self.num_clusters = num_clusters
        self.num_iterations = num_iterations
        self.num_dimensions = len(dataset.columns)
        self.dataset = dataset

    def initialize_centroids(self):
        inital_centroids = []
        i = 0

        self.add_centroid_column()

        while i < self.num_clusters:
            new_centroid = np.random.rand(self.num_dimensions)
            inital_centroids.append(new_centroid)
            i += 1

        self.centroids = inital_centroids
        print(self.centroids)

    def assign_instances_to_cluster(self):
        for index in self.dataset.index:
            nearest_centroid_index = self.get_nearest_centroid_index(dataset.iloc[[index]].to_numpy()[0][1:])
            dataset['Centroid'][index] = nearest_centroid_index
        return
    
    def calculate_clusters_center(self):
        i = 0

        while i < self.num_clusters:
            cluster_i = dataset[dataset['Centroid'] == i]
            centroid = np.mean(cluster_i, axis=0)
            self.centroids[i] = centroid.drop(columns=['Centroid']).to_numpy()
            i += 1
        
        print(self.centroids)
        return

    def get_nearest_centroid_index(self, instance):
        min_distance = INFINITE
        all_centroids = self.centroids
        nearest_centroid_index = 0

        for index, centroid in enumerate(all_centroids):
            current_distance = self.calculate_euclidian_distance(centroid, instance)
            if(current_distance < min_distance):
                min_distance = current_distance
                nearest_centroid_index = index

        return nearest_centroid_index

    def calculate_euclidian_distance(self, x1, x2):
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
    
    def add_centroid_column(self):
        num_rows = self.dataset.shape[0]
        formatted_dataset = self.dataset
        formatted_dataset['Centroid'] = np.full(num_rows, INVALID_CENTROID)
        self.dataset = formatted_dataset
    
    def calculate_avarage_of_vectors(self):
        return 
        


dataset = pd.read_csv(DATASET_PATH)

le = LabelEncoder()
dataset['Gender'] = le.fit_transform(dataset['Gender'])
# dataset[2:] = preprocessing.normalize(dataset[2:])
dataset = dataset.drop(columns=['CustomerID'])

kmeans = K_Means(NUM_CLUSTERS, dataset, NUM_ITERATIONS)
kmeans.initialize_centroids()
kmeans.assign_instances_to_cluster()
kmeans.calculate_clusters_center()
# print(kmeans.dataset)

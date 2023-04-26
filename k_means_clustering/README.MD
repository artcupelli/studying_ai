
# K-Means Clustering

### Clustering

Clustering is an unsupervised learning technique concerned with grouping instances into clusters. Instances in one cluster should be the most similar to each other and the most different to the ones in other clusters.

Personally, I'm studying this technique for the data mining field, but this also envolves applications in AI, process mining, and others.

Each cluster has a centroid, usually denoted with the ( **+** ) symbol. A centroid is the "center" of a individual cluster, usually computed as the average of the coordinates of all the instances inside a cluster.

Distance-based clustering algorithms assume a distance notation. One of the most popular way of achieving this technique is simply considering each instance as a *n*-dimensional vector and calculating the euclidian distance between each one .

### K-Means

**Pseudocode**

1. The algorithm begins with a random initialization of *k* centroids (a centroid being a *n*-dimension vector).
2. Using the choosen distance metric (Euclidian distance, Cosine similarity...), all instances are assigned to the closest centroid.
3. With the inital clusters formed, we compute the real center of each cluster, repositioning the centroids.
4. Again, all the instances are reassigned to the closest centroid.
5. Repeat steps 3 and 4

This heuristic does not guarantee that it finds the *k* clusters that minimize the avarage distance from an instance to its corresponding centroid. The result depends on the initalization, therefore it is a good practice to execute the algoithm a few times and select the best one.
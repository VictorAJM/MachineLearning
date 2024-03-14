import numpy as np
class Kmeans:
  def __init__(self,n_clusters):
    self.n_clusters = n_clusters
    self.centroids = None
    self.labels = None
  def fit(self, img):
    centroids = np.random.choice(img.shape[0], self.n_clusters, replace=False)
    initial_centroids = img[centroids]
    for _ in range(40):
      distances = np.linalg.norm(img[:, np.newaxis] - initial_centroids,axis=2)
      labels = np.argmin(distances, axis=1)
      for i in range(self.n_clusters):
        initial_centroids[i] = np.mean(img[labels == i], axis=0)
    self.centroids = initial_centroids
    self.labels = labels
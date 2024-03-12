from scipy import stats
import numpy as np
class nc: 
  def __init__(self):
    self.centroids = {}

  def fit(self,Xtrain,Ytrain):
    unique_targets = set(Ytrain)
    for target in unique_targets:
      target_samples = Xtrain[Ytrain == target]
      centroid = np.mean(target_samples,axis=0)
      self.centroids[target] = centroid
  def predict(self, Xtest):
    predicted_targets = []
    for sample in Xtest:
      distances = [np.linalg.norm(sample-centroid) for centroid in self.centroids.values()]
      predicted_target = list(self.centroids.keys())[np.argmin(distances)]
      predicted_targets.append(predicted_target)
    return predicted_targets

import numpy as np
from math import floor

def  euclidian_distance(x1, x2):
    x = []
    for i in range(len(x1)):
        x.append(x1.iloc[i])

    return np.sqrt(np.sum((x-x2)**2))

class OCC:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X):
        self.X_train = X
    
    def predict(self, X):
        predicted_labels = []
        for _, row in X.iterrows():
            predicted_labels.append(self._predict(row))

        return np.array(predicted_labels)

    def _predict(self, x):
        # evaluate the threshold
        # get the closest instance to the evaluated one
        delta, closest_to_x_index = self.get_closest(x)
        # get the distances of the k nearest neighbors  of the training set
        k_distances = self.get_k_distances(x, closest_to_x_index, self.k)
       
        # majority vote
        vote_target, vote_outlier = 0, 0
        # print("vote_target: ", vote_target)
        # print("vote_outlier: ", vote_outlier)
        for i in k_distances:
            if delta < i:
                vote_outlier += 1
            else:
                vote_target += 1
        # print("acabou")
                
        # Compute the distances
        if vote_target > vote_outlier:
            return "b'false'"
        else:
            return "b'true'"
    
    def get_closest(self, x):
        distances = []
        for _, row in self.X_train.iterrows():
            distances.append(euclidian_distance(x, row))

        index = np.argsort(distances)[0]
        return  distances[index], index

    def get_k_distances(self, x, index, k):
        distances = []
        for _, row in self.X_train.iterrows():
            distances.append(euclidian_distance(self.X_train.iloc[index], row))

        k_indices = np.argsort(distances) #indices dos k mais próximos
        k_indices = k_indices[1:self.k+1] # exclui o primeiro indice que eh o proprio elemento

        k_distances = []
        for index in k_indices:
            k_distances.append(distances[index])

        return k_distances
        

import numpy as np
from collections import Counter

def  euclidian_distance(x1, x2):
    x = []
    for i in range(len(x1)):
        x.append(x1.iloc[i])

    return np.sqrt(np.sum((x-x2)**2))

class WeightedKNN:

    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predicted_labels = []
        for _, row in X.iterrows():
            predicted_labels.append(self._predict(row))

        return np.array(predicted_labels)
    
    def _predict(self, x):
        # Compute the distances
        distances = []
        for _, row in self.X_train.iterrows():
            distances.append(euclidian_distance(x, row))


        # Get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k] #argsort returns the indices of the sorted array
        
        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]

        weights = []
        for i in k_indices:
            weights.append(1/(distances[i]**2))
        
        weight_true, weight_false = 0, 0
        for i in range(len(k_nearest_labels)):
            if k_nearest_labels[i] == b'false':
                weight_false += weights[i]
            elif k_nearest_labels[i] == b'true':
                weight_true += weights[i]
        

        # Biggest weight sum vote, most common class label
        if weight_true > weight_false:
            return b'true'
        else:
            return b'false'
        #return most_common[0][0]

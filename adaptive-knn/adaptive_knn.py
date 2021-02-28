import numpy as np
import random

def  euclidian_distance(x1, x2):
    x = []
    for i in range(len(x1)):
        x.append(x1.iloc[i])

    return np.sqrt(np.sum((x-x2)**2))

class AdaptiveKNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.radius = self.calculate_radius(self.X_train, self.y_train)

    def calculate_radius(self, X, Y, epsilon=0.00000000001):
        radius = []
        for i in range(len(X)):
            current_instance_radius = []
            for j in range(len(X)):
                if Y.iloc[i] != Y.iloc[j]:
                    distance = euclidian_distance(X.iloc[i], X.iloc[j]) - epsilon
                    if(distance >= 0):
                        current_instance_radius.append(distance)
            
            radius.append(min(current_instance_radius))
        
        return radius

    def predict(self, X):
        predicted_labels = []
        for _, row in X.iterrows():
            predicted_labels.append(self._predict(row))

        return np.array(predicted_labels)

    def _predict(self, x):
        OUTSIDE_CIRCLE = float('inf')
        
        # Compute distance-radius ratio
        distances = []
        for _, row in self.X_train.iterrows():
            distances.append(euclidian_distance(x, row))
        
        distance_radius_ratio = []
        for i in range(len(distances)):
            ratio = distances[i]/self.radius[i]
            if ratio > 1:
                distance_radius_ratio.append(OUTSIDE_CIRCLE)
            else:
                distance_radius_ratio.append(ratio)
        
        # Get k nearest samples, labels
        k_indices = np.argsort(distance_radius_ratio)[:self.k] #argsort returns the indices of the sorted array
        k_nearest_radius = [self.radius[i] for i in k_indices]

        true, false = 0, 0
        for i in range(len(k_indices)):
            if k_nearest_radius[i] == OUTSIDE_CIRCLE:
                pass
            elif self.y_train.iloc[k_indices[i]] == b'true':
                true += 1
            elif self.y_train.iloc[k_indices[i]] == b'false':
                false += 1

        if true > false:
            return b'true'
        elif false > true:
            return b'false'
        else:
            return b'true' if random.randint(1,100) % 2 == 0 else b'false'

from scipy.io import arff
import pandas as pd
from sklearn.model_selection import KFold
from knn import KNN
import numpy as np
from sklearn import preprocessing

#load CM1
data = arff.loadarff('./datasets/CM1.arff')

# separate data from labels
X = pd.DataFrame(data[0])
Y = X['defects']
X = X.drop(columns=['defects'])


# normalize data
arr = X.values
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(arr)
X = pd.DataFrame(X_scaled)


# create k-fold splits
kf = KFold(n_splits=10)

# instanciate classifier
clf = KNN(k=3)

index = 0
for train, test in kf.split(X):
    print('Running fold {}...'.format(index))

    clf.fit(X.iloc[train], Y.iloc[train])
    predictions = clf.predict(X.iloc[test])
    
    acc = (np.sum(predictions == Y.iloc[test]) / len(test)) * 100
    print("Accuracy for fold {}: {}%".format(index, acc))
    index += 1

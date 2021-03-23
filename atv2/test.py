import pandas as pd
from sklearn.model_selection import KFold
from knn import KNN
import numpy as np
from sklearn import preprocessing
import time
import csv

def normalize_data(X):
    arr = X.values
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(arr)
    return pd.DataFrame(X_scaled)

def build_dataframe(data):
    training_data = pd.DataFrame(data[0])
    labels = training_data['defects']
    training_data = training_data.drop(columns=['defects'])
    return training_data, labels


def main():
    K = [1,3]

    #load CM1
    data = pd.read_csv('./lvq_output/seed_137/kc1_lvq3.csv')

    X, Y = data.drop(columns=['defects']), data['defects']


    # normalize data
    #X = normalize_data(X)

    # create k-fold splits
    kf = KFold(n_splits=10)

    # instanciate classifier
    for k in K:
        clf = KNN(k=k)
        print("k equals {}".format(k))
        
        start_time = time.time()
        acc = []
        for train,test in kf.split(X):

            clf.fit(X.iloc[train], Y.iloc[train])
            predictions = clf.predict(X.iloc[test])
            acc.append((np.sum(predictions == Y.iloc[test]) / len(test)) * 100)
        
        end_time = time.time()
        
        acc = np.array(acc)
    
        print("mean accuracy: {}".format(np.mean(acc)))
        print("standard deviation: {}".format(np.std(acc)))
        print("time elapsed: {}".format(end_time - start_time))
    

if __name__ == '__main__':
    main()

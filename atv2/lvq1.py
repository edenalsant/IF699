import numpy as np
from scipy.io import arff
import pandas as pd
from random import randrange
import random as rand
from sklearn import preprocessing

def normalize_data(X):
    arr = X.values
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(arr)
    return pd.DataFrame(X_scaled)
def build_dataframe(data):
    training_data = pd.DataFrame(data[0])
    return training_data

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1-x2)**2)

def get_closest_neighbor(data, new_instance):
    distances = list()
    for _, d in data.iterrows():
        dist = euclidean_distance(d, new_instance)
        distances.append(dist)

    closest_neighbor_index = np.argsort(distances)[1]
    return data.iloc[closest_neighbor_index], closest_neighbor_index

def get_random_subset(X):

    rand.seed(137)
    first_row = X.loc[0]
    dataset_subset = X.iloc[[i for i in range(randrange(X.shape[0]))],
                            [i for i in range(len(first_row))]]

    labels = dataset_subset['defects']
    dataset_subset = dataset_subset.drop(columns=['defects'])
    return dataset_subset, labels
    

def lvq1(X, S, learning_rate, adjustment, y):
    rand.seed(None)
    while learning_rate > 0.01:
        p = X.iloc[randrange(X.shape[0])]
        p_label = p['defects']
        p = p[:-1]

        e, e_index = get_closest_neighbor(S, p)
        if y.iloc[e_index] != p_label:
            e = e + learning_rate * (p-e)
        else:
            e = e - learning_rate * (p-e)
           
        S.iloc[e_index] = e
        learning_rate = learning_rate * adjustment

    S['defects'] = y
    return S



def main():
    data = arff.loadarff('./datasets/KC1.arff')
    X = build_dataframe(data)
    X = X.sample(frac=1, random_state=20)
    X.reset_index(drop=True, inplace=True)
    Y = X['defects']
    X = X.drop(columns=['defects'])
    X = normalize_data(X)
    X['defects'] = Y


    Sx, Sy = get_random_subset(X)
    print(Sx.shape)
    learning_rate = 0.2
    adjustment = 0.9
    S = lvq1(X, Sx, learning_rate, adjustment, Sy)
    #S.to_csv('./lvq_output/seed_137/kc1_lvq1.csv')
    

main()

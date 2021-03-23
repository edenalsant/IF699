import numpy as np
from scipy.io import arff
import pandas as pd
from random import randrange
import random as rand
from sklearn import preprocessing
from lvq1 import lvq1
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning) 

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

def get_two_closest_neighbors(data, new_instance):
    distances = list()
    for _, d in data.iterrows():
        dist = euclidean_distance(d, new_instance)
        distances.append(dist)

    first_closest_neighbor_index = np.argsort(distances)[1]
    second_closest_neighbor_index = np.argsort(distances)[2]
    return data.iloc[first_closest_neighbor_index], data.iloc[second_closest_neighbor_index], first_closest_neighbor_index, second_closest_neighbor_index

def get_random_subset(X):

    rand.seed(98)
    first_row = X.loc[0]
    dataset_subset = X.iloc[[i for i in range(randrange(X.shape[0]))],
                            [i for i in range(len(first_row))]]

    labels = dataset_subset['defects']
    dataset_subset = dataset_subset.drop(columns=['defects'])
    return dataset_subset, labels

def is_at_window(x, ei, ej, w):
    dist_x_ei = euclidean_distance(x, ei)
    dist_x_ej = euclidean_distance(x, ej)
    s = (1-w)/(1+w)

    return True if min((dist_x_ei/dist_x_ej),(dist_x_ej/dist_x_ei)) > s else False

def lvq2(X, S, learning_rate, adjustment, y):
    rand.seed(None)
    while learning_rate > 0.01:
        p = X.iloc[randrange(X.shape[0])]
        p_label = p['defects']
        p = p[:-1]
        w = 0.5

        ei,ej, ei_index, ej_index = get_two_closest_neighbors(S, p)

        if is_at_window(p,ei,ej,w):
            if y.iloc[ei_index] != y.iloc[ej_index]:
                if y.iloc[ei_index] == p_label:
                    #faz conta
                    ei = ei + learning_rate * (p-ei)
                    ej = ej - learning_rate * (p-ej)
                else:
                    ei = ei - learning_rate * (p-ei)
                    ej = ej + learning_rate * (p-ej)
                    
        S.iloc[ei_index] = ei
        S.iloc[ej_index] = ej
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

    learning_rate = 0.2
    adjustment = 0.9

    S = lvq1(X, Sx, learning_rate, adjustment, Sy)
    Sy = S['defects']
    Sx = S.drop(columns=['defects'])
    S = lvq2(X, Sx, learning_rate, adjustment, Sy)
    S.to_csv('./lvq_output/seed_137/kc1_lvq2.csv')

main()

from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from math import floor
from occ import OCC
import time
import sys

def remove_outliers(X):
    lines_to_remove = []
    for index, row in X.iterrows():
        if row['defects'] == b'true':
            lines_to_remove.append(index)
    training_data = X.drop(lines_to_remove)
    outliers = X.iloc[lines_to_remove]

    return training_data, outliers

def trim_dataframe(df, usage_rate):
    total_rows = df.shape[0]
    number_training_rows = floor(total_rows*usage_rate)

    training_data = df.iloc[:number_training_rows, :]
    testing_data = df.iloc[number_training_rows:, :]

    return training_data, testing_data

def randomize_dataframe(df):
    df = df.sample(frac=1, random_state=20)
    df.reset_index(drop=True, inplace=True)
    return df

def build_dataframe(data, usage_rate):
    training_data = pd.DataFrame(data[0])
    training_data = randomize_dataframe(training_data)
    training_data, outliers = remove_outliers(training_data)
    training_data, test_data = trim_dataframe(training_data, usage_rate)
    return training_data, test_data, outliers

def normalize_data(X):
    arr = X.values
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(arr)
    return pd.DataFrame(X_scaled)

def drop_defects_column(dfs):
    final_df = []
    for df in dfs:
        final_df.append(df.drop(columns=['defects']))
    
    return final_df[0], final_df[1], final_df[2]


def main():
    data = arff.loadarff('./datasets/CM1.arff')
    k = int(sys.argv[1])
    usage_rate = float(sys.argv[2])
    #print("running with k = {} and usage rate = {}%".format(k, usage_rate*10))
    df, df_test, outliers = build_dataframe(data, usage_rate=usage_rate)
    df, df_test, outliers = drop_defects_column([df, df_test, outliers])

    df = normalize_data(df)
    df_test = normalize_data(df_test)
    outliers = normalize_data(outliers)

    clf = OCC(k=k)
    clf.fit(df)
    
    start_time = time.time()
    test_results = clf.predict(df_test) #todo mundo eh false
    end_time = time.time()
    counter = 0
    for t in test_results:
        if t == "b'false'":
            counter += 1
    print("k = {}".format(k))
    print("acerto no falsos: {}%".format(round((counter/len(test_results))*100, 2)))
    print("the results for tests were (expected all to be false):")
    print(test_results)
    print("time elapsed")
    print(end_time-start_time)
    print("")
    print("")
    start_time = time.time()
    outliers_results = clf.predict(outliers) #todo mundo eh true
    end_time = time.time()
    counter = 0
    for t in outliers_results:
       if t == "b'true'":
           counter += 1
    
    print("acerto no verdadeiros: {}%".format(round((counter/len(outliers_results)*100),2)))
    print("the results for outlies were (expected all to be true):")
    print(outliers_results)
    print("time elapsed")
    print(end_time-start_time)




main()

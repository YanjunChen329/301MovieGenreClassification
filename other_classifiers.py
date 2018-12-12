from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import pickle

# --------------------- Load Data ----------------------
all_train_df = pd.read_csv("train_data/train_data.csv")
train_df = pd.read_csv("train_data/train.csv")
validation_df = pd.read_csv("train_data/validation.csv")
all_train_df = shuffle(all_train_df)
# print(train_df)

all_train_data, all_train_label = all_train_df[["startYear", "runtimeMinutes", "IMDB Score"]].values, all_train_df[["Genre"]].values.ravel()
train_data, train_label = train_df[["startYear", "runtimeMinutes", "IMDB Score"]].values, train_df[["Genre"]].values.ravel()
valid_data, valid_label = validation_df[["startYear", "runtimeMinutes", "IMDB Score"]].values, validation_df[["Genre"]].values.ravel()
# print(all_train_data, all_train_label)

# --------------------- Bowen's Data ----------------------
# train_half = np.load("train_data/datou_train6.npy")
# train_data = np.concatenate((train_data, train_half), axis=1)
# SHUFFLE
# data_label = np.concatenate((train_data, np.reshape(train_label, (-1, 1))), axis=1)
# np.random.shuffle(data_label)
# train_data, train_label = data_label[:, :-1], data_label[:, -1]

# valid_half = np.load("train_data/datou_test6.npy")
# valid_data = np.concatenate((valid_data, valid_half), axis=1)

def preprocessing(df):
    year = df[["startYear"]].values - 1980 / 40.0
    runtime = df[["runtimeMinutes"]].values / 250.0
    score = df[["IMDB Score"]].values / 10.0
    return np.concatenate((year, runtime, score), axis=1)

def wrong_prediction_probability(prediction, truth):
    wrong = {0: 0, 1: 0, 2: 0, 3: 0}
    total = {0: 0, 1: 0, 2: 0, 3: 0}
    for i in range(prediction.shape[0]):
        print(prediction[i], truth[i])
        total[prediction[i]] += 1
        if prediction[i] != truth[i]:
            wrong[prediction[i]] += 1
    probability = {}
    probability[0] = wrong[0] / (1.0 * total[0])
    probability[1] = wrong[0] / (1.0 * total[1])
    probability[2] = wrong[0] / (1.0 * total[2])
    probability[3] = wrong[0] / (1.0 * total[3])
    return probability

# all_train_data = preprocessing(all_train_df)

alg = RandomForestClassifier(random_state=1, n_estimators=600, min_samples_split=4, min_samples_leaf=6)
# scores = cross_val_score(alg, all_train_data, all_train_label, cv=10)
alg.fit(train_data, train_label)
# print(scores, scores.mean())
# Predict on validation
# prediction = alg.predict(valid_data)
# print(wrong_prediction_probability(prediction, valid_label))
# print(np.sum((prediction == valid_label).astype(int)) / valid_label.shape[0]*1.0)
joblib.dump(alg, "normal_rf.pkl")

print()
alg2 = AdaBoostClassifier()
# scores2 = cross_val_score(alg2, all_train_data, all_train_label, cv=5)
# alg2.fit(train_data, train_label)
# print(scores2, scores2.mean())
# joblib.dump(alg, "adaboost.pkl")

print()
alg3 = KNeighborsClassifier(n_neighbors=9)
# scores3 = cross_val_score(alg3, all_train_data, all_train_label, cv=5)
# alg3.fit(train_data, train_label)
# print(scores3, scores3.mean())
# joblib.dump(alg3, "knn.pkl")

print()
alg4 = SVC(kernel='rbf', gamma='scale')
# scores4 = cross_val_score(alg4, all_train_data, all_train_label, cv=5)
# print(scores4, scores4.mean())
# alg4.fit(train_data, train_label)
# joblib.dump(alg4, "svm.pkl")

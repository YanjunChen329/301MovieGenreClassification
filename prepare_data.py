import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

path = "train_data/"

train_data = pd.read_csv("train_data/train_data.csv", header=0)
# print(train_data)
# thing = mpimg.imread("train_data/train_posters/111161.jpg")
# print(thing.shape)
# print(train_data)
# print(train_data[["imdbId", "Genre"]].values)
# print(train_data[["imdbId", "Genre"]].values.shape)

# --------------- Train / Test Split ----------------------
train, validation = train_test_split(train_data, train_size=0.8)
print(train)
print(validation)
train.to_csv(path + "train2.csv")
validation.to_csv(path + "validation2.csv")



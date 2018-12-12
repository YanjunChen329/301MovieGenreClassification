

"""
K/DA ELEC 301 FINAL Project Random Forest Classifier Part.
Author: Bowen Liu
E - Mail: libowenbob@hotmail.com
"""


import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.image as mpimg


trainning_read = pandas.read_csv("train_data.csv")

train_data_collection = trainning_read.values
data_length = len(train_data_collection)
real_test_read = pandas.read_csv("test_data.csv")
real_test_collection = real_test_read.values





def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])





def read_poster(id):
    s = "train_posters/" + str(id) + ".jpg"
    img = mpimg.imread(s)
    gray = rgb2gray(img)
    return gray.reshape(data_length)


def read_test_poster(id):
    s = "test_posters/" + str(id) + ".jpg"
    img = mpimg.imread(s)
    gray = rgb2gray(img)
    return gray.reshape(data_length)

def read_title_length(data_collection):
    length = len(data_collection)
    title_length = np.empty((length, 1))
    for i in range(length):
        title_length[i] = len(data_collection[i][2].split(" "))
    return title_length



def images_matrix(data_collection):
    length = len(data_collection)
    image_matrix = np.empty((length, data_length))
    for i in range(length):
        image_poster_grey = read_poster(data_collection[i][1])
        image_matrix[i] = image_poster_grey
    return image_matrix

def test_image_matrix(data_collection):
    length = len(data_collection)
    image_matrix = np.empty((length, data_length))
    for i in range(length):
        image_poster_grey = read_test_poster(data_collection[i][1])
        image_matrix[i] = image_poster_grey
    return image_matrix


def give_divid(images_m,number):
    length = len(images_m)
    maxgray_list = images_m.max(axis=1)
    maxgray = maxgray_list.max()
    mingray_list = images_m.min(axis=1)
    mingray = mingray_list.min()
    divid = (maxgray-mingray)/number
    matrix_16 = np.zeros((length, number))
    for i in range(length):
        for j in range(0,number):

            matrix_16[i][j] = (images_m[i] >= divid * j).sum() - (images_m[i] >= divid * (j+1)).sum()
    return matrix_16


def grey_averange(images_m):
    length = len(images_m)
    mean_gray = np.zeros((length, 1))
    for i in range(length):

       mean_gray[i][0] = np.average(images_m[i])
    return mean_gray


def process_data_main():
    images_m = test_image_matrix(real_test_collection)
    np.save("image_matrix",images_m)
    test_6 = give_divid(images_m,16.0)
    np.save("6_matrix", test_6)
    test_grey_avg = grey_averange(images_m)
    np.save("training_grey_avg", test_grey_avg)

    images_m = test_image_matrix(real_test_collection)
    np.save("test_image_matrix",images_m)
    test_6 = give_divid(images_m,16.0)
    np.save("test_16", test_6)
    test_grey_avg = grey_averange(images_m)
    np.save("test_grey_avg", test_grey_avg)



def with_color_density(data,s):

    s  = "color_desity_" + s +".npy"
    a = np.load(s)
    # b = np.concatenate((data, a[:, 0:3]), axis=1)
    c = np.concatenate((data, a), axis=1)
    return c



def read_train_data(color,title):
    image_matrix = np.load("6_matrix.npy")


    train_data_with_image = np.concatenate((train_data_collection,image_matrix), axis=1)
    if title:
        title_length = read_title_length(train_data_collection)
        train_data_with_image = np.concatenate((train_data_with_image, title_length), axis=1)
    if color:
        train_data_with_image = with_color_density(train_data_with_image,"train")
    np.random.shuffle(train_data_with_image)
    train_data_stat = train_data_with_image[:, 3:6]
    train_data_im = train_data_with_image[:,7:]

    train_data = np.concatenate((train_data_stat,train_data_im), axis=1)
    # train_data = train_data_with_image[:,7:]
    train_label = train_data_with_image[:, 6:7]
    train_data, test_data, train_label, test_label = train_test_split(train_data, train_label, test_size=0.10,
                                                                      random_state=0)
    train_label = np.reshape(train_label, len(train_label))
    train_label = train_label.tolist()
    test_label = np.reshape(test_label,len(test_label))
    test_label = test_label.tolist()

    return train_data, test_data, train_label, test_label


def read_train_without_split(color,title):
    image_matrix = np.load("6_matrix.npy")



    train_data_with_image = np.concatenate((train_data_collection,image_matrix), axis=1)

    if title:
        title_length = read_title_length(train_data_collection)
        train_data_with_image = np.concatenate((train_data_with_image, title_length), axis=1)

    if color:
        train_data_with_image = with_color_density(train_data_with_image,"train")
    np.random.shuffle(train_data_with_image)
    train_data_stat = train_data_with_image[:, 3:6]
    train_data_im = train_data_with_image[:,7:]

    train_data = np.concatenate((train_data_stat,train_data_im), axis=1)
    train_label = train_data_with_image[:, 6:7]
    train_label = np.reshape(train_label, len(train_label))
    train_label = train_label.tolist()

    return train_data,train_label


def accuracy(predictions,labels):

    error_dict = {0:{1:0,2:0,3:0}, 1:{0:0,2:0,3:0},2:{1:0,0:0,3:0},3:{0:0,2:0,1:0}}

    count = 0.0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            count = count + 1.0
        else:
            error_dict[labels[i]][predictions[i]] += 1

    # print(error_dict)
    accuracy = count / len(predictions)
    print("--- Accuracy value is " + str(accuracy))
    return accuracy


def give_real_test(color,title):
    test_grey = np.load("test_16.npy")

    real_test = np.concatenate((real_test_collection, test_grey), axis=1)
    real_test = real_test[:,3:]
    if title:
        title_length = read_title_length(real_test_collection)
        real_test = np.concatenate((real_test, title_length), axis=1)

    if color:
        real_test = with_color_density(real_test,"test")
    return real_test



# print(test_data[0])

def fit_all(color,title_length,random_state, n_estimators, min_samples_split, min_samples_leaf):
    # print("#################f fit all with title length and 16 division#########################")
    train_data, train_label = read_train_without_split(color,title_length)
    real_test_data = give_real_test(color, title_length)
    alg = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    alg.fit(train_data, train_label)
    # Compute the accuracy score for all the cross validation folds.
    # train_data = np.concatenate((train_data,test_data),axis = 0)
    # train_label = np.concatenate((train_label,test_label),axis=0)
    scores = cross_val_score(alg, train_data, train_label, cv=10)

    # # Take the mean of the scores (because we have one for each fold)
    # print(scores)
    # print("Cross validation scores = " + str(scores.mean()))

    full_predictions = []
    # Fit the algorithm using the full training data.
    # alg.fit(train_data, train_label)
    # Predict using the test dataset.
    predictions = alg.predict_proba(train_data).astype(float)
    predictions = predictions.argmax(axis=1)
    # print("Training Accuracy")
    # predictions = alg.predict_proba(real_test_data).astype(float)
    # predictions = predictions.argmax(axis=1)
    # np.savetxt("2018_12_9_5.csv",predictions)
    return accuracy(predictions,train_label), scores.mean()



def fit_partial(color,title_length,random_state, n_estimators, min_samples_split, min_samples_leaf):
    # print("#################f fit partially with 16 division#########################")
    train_data, test_data, train_label, test_label = read_train_data(color,title_length)
    real_test_data = give_real_test(color, title_length)
    alg = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    scores = cross_val_score(alg, train_data, train_label, cv=5)
    # print(scores)
    # print("Cross validation scores = " + str(scores.mean()))
    alg.fit(train_data, train_label)
    predictions = alg.predict_proba(test_data).astype(float)
    predictions = predictions.argmax(axis=1)
    # print("Validation Accuracy")
    # predictions = alg.predict_proba(real_test_data).astype(float)
    # predictions = predictions.argmax(axis=1)
    # np.savetxt("2018_12_9_4.csv",predictions)
    return accuracy(predictions,test_label), scores.mean()



def see_average(color,title,number):

    train_accuracy = 0
    train_cross_score = 0
    local_validation_accuracy = 0
    local_cross_score = 0
    for i in range(number):
        a,b = fit_all(color,title,1, 600, 10,5)
        c,d = fit_partial(color,title,1,600,10,5)
        train_accuracy += a
        train_cross_score += b
        local_validation_accuracy += c
        local_cross_score += d

    train_accuracy /= number
    train_cross_score /= number
    local_validation_accuracy /= number
    local_cross_score /= number

    print("########## Averaging Training Accuracy and Cross Scores ##################")
    print(train_accuracy,train_cross_score)
    print("########## Averaging Local Validation Accuracy and Local Cross Scores ##################")
    print(local_validation_accuracy,local_cross_score)


see_average(True,False,10)






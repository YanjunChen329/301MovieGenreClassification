import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pandas as pd
import numpy as np

import Resnet
import movie_dataset as md
import pickle

BATCH_SIZE = 32  # number of images for each epoch
GPU_IN_USE = True  # whether using GPU

# ===========================================================
# Prepare test dataset
# ===========================================================
print("***** prepare data ******")
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
test_set = md.MovieDataSet("test_data/test_data.csv", training=False, transform=transform,
                           poster_path="./test_data/test_posters/{}.jpg", poster_data_path="./test_data/posters_data/{}")
test_dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE)

validation_set = md.MovieDataSet("train_data/validation.csv", training=True, transform=transform)
validation_dataloader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=BATCH_SIZE)

# ===========================================================
# Prepare dataset for ensembles
# ===========================================================
print("***** prepare ensemble data ******")
validation_df = pd.read_csv("train_data/validation.csv")
valid_data, valid_label = validation_df[["startYear", "runtimeMinutes", "IMDB Score"]].values, validation_df[["Genre"]].values.ravel()
# valid_half = np.load("train_data/datou_test6.npy")
# valid_data = np.concatenate((valid_data, valid_half), axis=1)
test_df = pd.read_csv("test_data/test_data.csv")
test_data = test_df[["startYear", "runtimeMinutes", "IMDB Score"]].values


def validate(model, ep=0):
    total = 0
    valid_correct = 0
    model.eval()
    prediction = np.array([])
    wrong_prediction = np.array([])
    wrong_truth = np.array([])

    for iteration, ((x, info), y) in enumerate(validation_dataloader):
        x = Variable(x).cuda() if GPU_IN_USE else Variable(x)
        info = Variable(info).cuda() if GPU_IN_USE else Variable(info)
        y = Variable(y).cuda() if GPU_IN_USE else Variable(y)

        output = model(x, info)
        _, predicted = torch.max(output.data, 1)
        prediction = np.concatenate((prediction, predicted.cpu().numpy().astype(int)), axis=None)


        # wrong_prediction = np.concatenate((wrong_prediction, predicted[predicted != y.data]), axis=None)
        # wrong_truth = np.concatenate((wrong_truth, y.data[predicted != y.data]), axis=None)

        valid_correct += (predicted == y.data).sum().item()
        total += y.data.shape[0]
    print("epoch: {}; Accuracy: {}".format(ep, valid_correct/float(total)))
    # wrong = np.concatenate((np.reshape(wrong_prediction, (-1, 1)), np.reshape(wrong_truth, (-1, 1))), axis=1)
    # np.savetxt("predict_wrong.txt", wrong)
    return prediction


def predict(model, fname):
    prediction = np.array([], dtype=int)
    model.eval()

    for iteration, (x, info) in enumerate(test_dataloader):
        x = Variable(x).cuda() if GPU_IN_USE else Variable(x)
        info = Variable(info).cuda() if GPU_IN_USE else Variable(info)
        output = model(x, info)
        _, predicted = torch.max(output.data, 1)
        prediction = np.concatenate((prediction, predicted.cpu().numpy().astype(int)), axis=None)

    df = pd.DataFrame(data=prediction)
    df["Id"] = df.index
    df.columns = ["Category", "Id"]
    df = df.reindex(columns=["Id", "Category"])
    print(df)
    df.to_csv(fname, index=False)
    # return prediction


def test_validate():
    for i in range(25, 50):
        model = Resnet.resnet34(num_classes=4)
        if GPU_IN_USE:
            model.cuda()
        model.load_state_dict(torch.load("model_state/state34_ep{}".format(i)))
        validate(model, i)


def wrong_prediction_probability(prediction, truth):
    wrong = {0: 0, 1: 0, 2: 0, 3: 0}
    total = {0: 0, 1: 0, 2: 0, 3: 0}
    for i in range(prediction.shape[0]):
        total[prediction[i]] += 1
        if prediction[i] != truth[i]:
            wrong[prediction[i]] += 1
    probability = {}
    probability[0] = wrong[0] / (1.0 * total[0])
    probability[1] = wrong[0] / (1.0 * total[1])
    probability[2] = wrong[0] / (1.0 * total[2])
    probability[3] = wrong[0] / (1.0 * total[3])
    return probability


def ensemble_validate(rf):
    rf_result = rf.predict(valid_data)
    nn_result = np.loadtxt("temp.txt")
    print(rf_result.shape, nn_result.shape)
    rf_p = wrong_prediction_probability(rf_result, valid_label)
    nn_p = wrong_prediction_probability(nn_result, valid_label)
    print(rf_p)
    print(nn_p)
    ensemble_prediction = np.zeros(rf_result.shape[0])

    both_wrong = 0
    for i, lst in enumerate(zip(rf_result, nn_result)):
        if lst[0] == lst[1]:
            result = lst[0]
        else:
            p_0true = 1 - rf_p[lst[0]]
            p_1true = 1 - nn_p[lst[1]]
            prob = p_0true / (p_1true + p_0true)
            rand = np.random.rand()
            if rand < prob:
                result = lst[0]
            else:
                result = lst[1]
            # result = lst[1]
            # result = lst[1] if rf_p[lst[0]] >= nn_p[lst[1]] else lst[0]
        # check how many both wrong
        if lst[0] != valid_label[i] and lst[1] != valid_label[i]:
            both_wrong += 1
        # print(i, lst, result, valid_label[i])
        ensemble_prediction[i] = result

    print(both_wrong)
    print(np.sum((ensemble_prediction == valid_label).astype(int)) / float(valid_label.shape[0]))


def ensemble_test(rf, nn_result, fname):
    rf_valid = rf.predict(valid_data)
    nn_valid = np.loadtxt("temp.txt")
    print(rf_valid.shape, nn_valid.shape)
    rf_p = wrong_prediction_probability(rf_valid, valid_label)
    nn_p = wrong_prediction_probability(nn_valid, valid_label)

    rf_result = rf.predict(test_data)
    ensemble_prediction = np.zeros(rf_result.shape[0])

    for i, lst in enumerate(zip(rf_result, nn_result)):
        if lst[0] == lst[1]:
            result = lst[0]
        else:
            p_0true = 1 - rf_p[lst[0]]
            p_1true = 1 - nn_p[lst[1]]
            prob = (p_0true-0.1) / (p_1true + p_0true)
            rand = np.random.rand()
            if rand < prob:
                result = lst[0]
            else:
                result = lst[1]
                # result = lst[1]
                # result = lst[1] if rf_p[lst[0]] >= nn_p[lst[1]] else lst[0]
        print(i, lst, result)
        ensemble_prediction[i] = result

    df = pd.DataFrame(data=ensemble_prediction)
    df["Id"] = df.index
    df.columns = ["Category", "Id"]
    df = df.reindex(columns=["Id", "Category"])
    print(df)
    df.to_csv(fname, index=False)


if __name__ == '__main__':
    print("***** prepare model ******")
    # model = Resnet.resnet18(num_classes=4)
    # if GPU_IN_USE:
    #     model.cuda()
    # model.load_state_dict(torch.load("model_state/stateEX2_ep39"))

    # test_validate()
    # predict(model, "prediction5.csv")
    # validate(model, 25)

    rf = joblib.load("normal_rf.pkl")
    # nn_result = validate(nn)
    # nn_result = pd.read_csv("prediction2.csv").values[:, 1].ravel()
    # np.savetxt("temp.txt", nn_result)
    ensemble_validate(rf)
    # ensemble_test(rf, nn_result, "prediction4.csv")

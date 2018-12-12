import torch
from torch.utils import data
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time


class MovieDataSet(data.Dataset):
    def __init__(self, data_path, training=False, transform=transforms.ToTensor(),
                 poster_path="./train_data/train_posters/{}.jpg", poster_data_path="./train_data/posters_data/{}.npy"):
        self.data_path = data_path
        self.data = pd.read_csv(data_path, header=0)
        self.id_list = self.data[["imdbId", "Genre"]].values if training else self.data[["imdbId"]].values
        startYear = (self.data[["startYear"]].values - 1980) / 40.0
        runtime = self.data[["runtimeMinutes"]].values / 250.0
        score = self.data[["IMDB Score"]].values / 10.0
        titles = self.data[["primaryTitle"]].values
        titleLength = np.apply_along_axis(lambda x: len(x[0]), 1, titles).reshape((-1, 1)) / 100.0
        titleWordNum = np.apply_along_axis(lambda x: len(x[0].split(" ")), 1, titles).reshape((-1, 1)) / 20.0
        # print(titleLength, titleWordNum)
        self.extra_info = np.concatenate((startYear, runtime, score, titleLength, titleWordNum), axis=1)
        self.extra_info = torch.from_numpy(self.extra_info).type(torch.cuda.FloatTensor)

        self.training = training
        self.transform = transform
        self.poster_path = poster_path
        self.poster_data_path = poster_data_path

    def __len__(self):
        return self.id_list.shape[0]

    def __getitem__(self, index):
        imdbID = self.id_list[index, 0]
        extra_info = self.extra_info[index, :]
        try:
            poster = np.load(self.poster_data_path.format(imdbID))
            poster = torch.from_numpy(poster)
        except IOError:
            poster_data = mpimg.imread(self.poster_path.format(imdbID))
            poster = poster_data
            poster = self.transform(poster)
            np.save(self.poster_data_path.format(imdbID), poster)

        if self.training:
            return (poster, extra_info), self.id_list[index, 1]
        else:
            return poster, extra_info
    
    def show_poster(self, imdbID):
        poster_data = mpimg.imread(self.poster_path.format(imdbID))
        plt.imshow(poster_data)
        plt.show()


if __name__ == '__main__':
    dataset = MovieDataSet("./train_data/train_data.csv", training=True)
    start = time.time()
    print(dataset[0][0])
    print(time.time() - start)

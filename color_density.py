

"""
Color Density Module
Author: Bowen Liu
E - Mail: libowenbob@hotmail.com
"""
import pandas
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mpl_toolkits.mplot3d.axes3d as p3
import colorsys

trainning_read = pandas.read_csv("train_data.csv")
train_data_collection = trainning_read.values
real_test_read = pandas.read_csv("test_data.csv")
real_test_collection = real_test_read.values


def colordensity(path,show):
    img = mpimg.imread(path)
    if len(img.shape) <= 2:
        print(path,"!!!!!!!")
        return [0, 0, 0, 0, 0, 0,0, 0, 0]
    elif img.shape[2] == 4:
        print(path,"RGBA!!!!!!")
        return [0, 0, 0, 0, 0, 0,0, 0, 0]
    # (2) Get image width & height in pixels

    [xs, ys,zs] = img.shape
    max_intensity = 100
    hues = {}

    # (3) Examine each pixel in the image file
    for x in range(0, xs):
        for y in range(0, ys):
            # (4)  Get the RGB color of the pixel
            [r, g, b] = img[x, y,:]

            # (5)  Normalize pixel color values
            r /= 255.0
            g /= 255.0
            b /= 255.0

            # (6)  Convert RGB color to HSV
            [h, s, v] = colorsys.rgb_to_hsv(r, g, b)

            # (7)  Marginalize s; count how many pixels have matching (h, v)
            if h not in hues:
                hues[h] = {}
            if v not in hues[h]:
                hues[h][v] = 1
            else:
                if hues[h][v] < max_intensity:
                    hues[h][v] += 1
    # (8)   Decompose the hues object into a set of one dimensional arrays we can use with matplotlib
    h_ = []
    v_ = []
    i = []
    colours = []

    for h in hues:
        for v in hues[h]:
            h_.append(h)
            v_.append(v)
            i.append(hues[h][v])
            [r, g, b] = colorsys.hsv_to_rgb(h, 1, v)
            colours.append([r, g, b])

    if show:
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.scatter(h_, v_, i, s=5, c=colours, lw=0)

        ax.set_xlabel('Hue')
        ax.set_ylabel('Value')
        ax.set_zlabel('Intensity')
        fig.add_axes(ax)
        plt.show()


    max_ind = i.index(max(i))

    [r, g, b] = colorsys.hsv_to_rgb(h_[max_ind], 1, v_[max_ind])

    next = [r, g, b]
    while np.linalg.norm(np.subtract([r,g,b],next)) < 0.2:
        i.pop(max_ind)
        h_.pop(max_ind)
        v_.pop(max_ind)
        max_ind = i.index(max(i))
        [rt, gt, bt] = colorsys.hsv_to_rgb(h_[max_ind], 1, v_[max_ind])
        next = [rt, gt, bt]

    [r1,g1,b1] = next

    while np.linalg.norm(np.subtract([r1,g1,b1],next)) < 0.08 or np.linalg.norm(np.subtract([r,g,b],next)) < 0.08:
        i.pop(max_ind)
        h_.pop(max_ind)
        v_.pop(max_ind)
        max_ind = i.index(max(i))
        [rt, gt, bt] = colorsys.hsv_to_rgb(h_[max_ind], 1, v_[max_ind])
        next = [rt, gt, bt]

    [r2, g2, b2] = next

    return [r, g, b ,r1, g1, b1 , r2,g2,b2]

def see_color_density(path):
    [r,g,b,r1,g1,b1,r2,g2,b2] = colordensity(path, True)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    rect = plt.Rectangle((0.1, 0.1), 0.25, 0.25,color=(r,g,b))
    ax.add_patch(rect)
    rect = plt.Rectangle((0.35, 0.35), 0.25, 0.25,color=(r1,g1,b1))
    ax.add_patch(rect)
    rect = plt.Rectangle((0.6, 0.6), 0.25, 0.25,color=(r2,g2,b2))
    ax.add_patch(rect)

    plt.show()
print(see_color_density('train_posters/116136.jpg'))

def get_nb_rgb():
    rgbs = []
    for i in range(len(train_data_collection)):
        id = train_data_collection[i][1]
        s = "train_posters/" + str(id) + ".jpg"
        rgbs.append(colordensity(s,False))
    np.save("color_desity_train",rgbs)

    rgbs = []
    for i in range(len(real_test_collection)):
        id = real_test_collection[i][1]
        s = "test_posters/" + str(id) + ".jpg"
        rgbs.append(colordensity(s, False))
    np.save("color_desity_test",rgbs)

# get_nb_rgb()
a = np.load("color_desity_test.npy")










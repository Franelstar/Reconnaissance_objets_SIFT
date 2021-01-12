import cv2 as cv
import sys
import argparse

import matplotlib
import notebook as notebook
from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np
from math import *
import os
from mpl_toolkits import mplot3d
from fonctions_prediction_evaluation import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--threshold", required=False, help="Detection threshold (Default 0.6)", default=0.6)
    parser.add_argument("-K", "--neighbors", required=False, help="Number of neighbors (Default 5)", default=5)
    parser.add_argument("-i", "--image", required=True, help="Path to the image to predict")
    parser.add_argument("-d", "--data", required=False, help="Dataset", default=1)

    args = vars(parser.parse_args())

    image_test = args["image"]
    threshold = args["threshold"]
    neighbors = args["neighbors"]
    data = args["data"]
    if int(data) != 1 and int(data) != 2:
        data = 1

    try:
        print(colored('* Loading resources ...', attrs=['dark']))
        if int(data) == 1:
            descriptors_train = np.load('model/descripteurs.npy', allow_pickle=True)
            label_train = np.load('model/labels.npy', allow_pickle=True)
            images_train = np.load('model/images.npy', allow_pickle=True)
            k_points = np.load('model/k_points.npy', allow_pickle=True)
        else:
            descriptors_train = np.load('model/descripteurs_2.npy', allow_pickle=True)
            label_train = np.load('model/labels_2.npy', allow_pickle=True)
            images_train = np.load('model/images_2.npy', allow_pickle=True)
            k_points = np.load('model/k_points_2.npy', allow_pickle=True)
        for i1, kp_t in enumerate(k_points):
            if type(kp_t) is tuple:
                k_points[i1] = cv.KeyPoint(x=kp_t[0][0], y=kp_t[0][1], _size=kp_t[1], _angle=kp_t[2],
                                           _response=kp_t[3], _octave=kp_t[4], _class_id=kp_t[5])
            else:
                for i2, kp_ in enumerate(kp_t):
                    if type(kp_) is tuple:
                        k_points[i1][i2] = cv.KeyPoint(x=kp_[0][0], y=kp_[0][1], _size=kp_[1], _angle=kp_[2],
                                                       _response=kp_[3], _octave=kp_[4], _class_id=kp_[5])
                    else:
                        for i3, kp in enumerate(kp_):
                            if type(kp) is tuple:
                                k_points[i1][i2][i3] = cv.KeyPoint(x=kp[0][0], y=kp[0][1], _size=kp[1],
                                                                   _angle=kp[2], _response=kp[3], _octave=kp[4],
                                                                   _class_id=kp[5])
        print(colored('* Loading resources complete', 'green'))
        print()

    except:
        print(colored('Some files are missing', 'red'))
        sys.exit(0)

    image = cv.imread(image_test)

    # Grayscale change
    print(colored('* Grayscale change ...', attrs=['dark']))
    try:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_test_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    except:
        print(colored("The path is not good", 'red'))
        sys.exit(0)
    print(colored('* Grayscale change complete', 'green'))
    print()

    # Generate SIFT features
    print(colored('* Generate SIFT features ...', attrs=['dark']))
    sift = cv.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image_test_gray, None)
    print(colored('* Generate SIFT features complete', 'green'))
    print()

    # Make prediction
    print(colored('* Make prediction ...', attrs=['dark']))
    result = predict_desc(des, descriptors_train, label_train, threshold, neighbors)
    print(colored('* Make prediction complete', 'green'))
    print()

    print(colored('* The predicted object is : {}'.format(result[0]), 'green'))
    print(colored('* The score is : {}'.format(result[1]), 'green'))

    # We check if the training image is available
    try:
        out = cv.drawMatches(image, kp, images_train[result[3]], k_points[result[3]], result[2], None)
        print(colored('Press the 0 key to close the image', attrs=['dark']))
        print()

        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(3, 2, 1)
        plt.imshow(image)
        plt.title('Image to predict')
        plt.axis("off")

        fig.add_subplot(3, 2, 2)
        plt.imshow(images_train[result[3]])
        plt.title('Corresponding image : {}'.format(result[0]))
        plt.axis("off")

        fig.add_subplot(3, 2, 3)
        plt.imshow(cv.drawKeypoints(image, kp, image, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
        plt.title('Key points')
        plt.axis("off")

        fig.add_subplot(3, 2, 4)
        plt.imshow(cv.drawKeypoints(images_train[result[3]], k_points[result[3]], images_train[result[3]],
                                    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
        plt.title('Key points')
        plt.axis("off")

        fig.add_subplot(3, 1, 3)
        plt.imshow(out)
        plt.title('Matching ({} corresponding)'.format(len(result[2])))
        plt.axis("off")

        plt.savefig('result.png')

        # cv.imshow('Detection result', cv.resize(image, (600, 600), interpolation=cv.INTER_AREA))
        cv.imshow('Predicted {}'.format(result[0]), cv.resize(cv.imread('result.png')
                                                              , (600, 600), interpolation=cv.INTER_AREA))
        cv.waitKey(0)
        cv.destroyAllWindows()
    except:
        pass


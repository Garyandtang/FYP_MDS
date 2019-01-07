# # sum = 0
# # for a in range(10):
# #     b = a+1
# #     filename = "%d.tex" % (b+1)
# #     print(filename)
# # #
# # #
# import cv2
# img = cv2.imread("1.pgm",0)
# # cv2.imshow("a", img)
# # cv2.waitKey(0)
# cv2.imwrite("./data/sss.jpg", img)


import csv
import cv2
import os
# define a fixed list images = []*400
#if you want to use index: for i, item in enumerate(reader):
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as pl
training_num = 3



#get the high resolution images from csv file
high_training_images = []
high_training_labels = []
high_testing_images = []
high_testing_labels = []

csvFile = open("high_at.csv", "r")
reader = csv.reader(csvFile)
for i, item in enumerate(reader):
    if i % 10 < training_num:
        high_training_images.append(item[0])
        high_training_labels.append(item[1])
    else:
        high_testing_images.append(item[0])
        high_testing_labels.append(item[1])
csvFile.close()

#get the low resolution images from csv file
low_training_images = []
low_training_labels = []
low_testing_images = []
low_testing_labels = []

csvFile = open("low_at.csv", "r")
reader = csv.reader(csvFile)
for i, item in enumerate(reader):
    if i % 10 < training_num:
        low_training_images.append(item[0])
        low_training_labels.append(item[1])
    else:
        low_testing_images.append(item[0])
        low_testing_labels.append(item[1])
csvFile.close()


#Mnumpy for storing the data  training
high_img = Image.open(high_training_images[0])
high_weight, high_height = high_img.size
high_size = high_weight * high_height
low_img = Image.open((low_training_images[0]))
low_size = low_img.size[0]*low_img.size[1]
high_resolution_training_set = np.zeros(shape=(len(high_training_images), high_size))
low_resolution_training_set = np.zeros(shape=(len(low_training_images), low_size))
print(high_resolution_training_set)
for i,  item in enumerate(high_training_images):
    img = Image.open(item)
    img = np.array(img).reshape(high_size)
    high_resolution_training_set[i] = img
for i, item in enumerate(low_training_images):
    img = Image.open(item)
    img = np.array(img).reshape(low_img.size[0]*low_img.size[1])
    low_resolution_training_set[i] = img


#numpy for storing the data testing
high_resolution_testing_set = np.empty((0, high_size))
low_resolution_testing_set = np.empty((0, low_img.size[0]*low_img.size[1]))
for item in high_testing_images:
    img = Image.open(item)
    img = np.array(img).reshape(high_size)
    high_resolution_testing_set = np.vstack((high_resolution_testing_set, img))
for item in low_testing_images:
    img = Image.open(item)
    img = np.array(img).reshape(low_img.size[0]*low_img.size[1])
    low_resolution_testing_set = np.vstack((low_resolution_testing_set, img))


# 零均值化
def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    newData = dataMat - meanVal
    return newData, meanVal


def pca(dataMat, n):
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
    n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
    lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
    return lowDDataMat

a = np.array([[2,3,2], [1, 2,3], [3, 4, 4],[3, 6, 5]])
print(a)
#
# xx, yy = zeroMean(high_resolution_testing_set)
# print(high_resolution_training_set)
# print(xx[1].size)
a = np.ones([5,2])

x = np.cov(high_resolution_training_set, rowvar=0)
eigVals, eigVects = np.linalg.eig(np.mat(x))
print(eigVals)

# # transform image to one dimensional array
# a = '1.pgm'
# img = Image.open(a)
#
# x = img.load()
# weight, height = img.size
# size = weight * height
# b = np.array(img).reshape(size)
#
# #vstack or hstack
# f = np.empty((0, size))
# list = np.vstack((b, b))
# f = np.vstack((f, list))
# f = np.vstack((f, list))
# print(f[0, :])
#

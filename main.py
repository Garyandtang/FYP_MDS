import csv
import cv2
import numpy as np
from PIL import Image

# define a fixed list images = []*400
#if you want to use index: for i, item in enumerate(reader):

#Paremeters:
training_num = 1   #number of images per subject that is using for training (both high resolution and low resolution)
common_space_dim = 50
landa = 0.5

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
high_resolution_training_set = np.empty((0, high_size))
low_resolution_training_set = np.empty((0, low_img.size[0]*low_img.size[1]))
for item in high_training_images:
    img = Image.open(item)
    img = np.array(img).reshape(high_size)
    high_resolution_training_set = np.vstack((high_resolution_training_set, img))
for item in low_training_images:
    img = Image.open(item)
    img = np.array(img).reshape(low_img.size[0]*low_img.size[1])
    low_resolution_training_set = np.vstack((low_resolution_training_set, img))


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


# # calculate the HR images distances   PAY ATTENTION: the method to calculate the distance need to be improved
# # L1-metric
# testing_num = 10 - training_num
# d = np.zeros(40) #HR images distance of 40 subjects
# for i in range(40):
#     dist = 0
#     for j in range(testing_num - 1):
#         dist = np.linalg.norm(high_resolution_training_set[i, :] - high_resolution_training_set[i+1+j, :]) + dist
#     d[i] = dist / (testing_num-1)
#

# Initialize the W and get the cost function J_W
Wh = np.ones((common_space_dim, high_size))
Wl = np.ones((common_space_dim, low_size))
W = np.hstack((Wl, Wh))   #50 * l+h
indexx = 0
indexyy = 0
J_W = 0
# for h_i, h_img in enumerate(high_resolution_training_set):
#     for l_i, l_img in enumerate(low_resolution_training_set):
#         combine_t = np.array([np.hstack((l_img, -h_img))])
#         if high_testing_labels[h_i] == low_training_labels[l_i]:
#             a = (1 - landa) + landa
#             x = a * combine_t.transpose().dot(combine_t)
#
#         else:
#             a = landa
#             x = a * combine_t.transpose() * combine_t
#         indexx = indexx +1
#         print(x)
#         print(indexx)
#         J_W = J_W + x

#
# J_W = 0
# for h_i, h_img in enumerate(high_resolution_training_set):
#     for l_i, l_img in enumerate(low_resolution_training_set):
#         combine_t = np.array([np.hstack((l_img, -h_img))])
#         combine = combine_t.transpose()
#         if high_testing_labels[h_i] == low_training_labels[l_i]:
#             indexx = indexx + 1
#             a = (1 - landa) + landa
#             b = landa/a
#             d = np.linalg.norm(high_resolution_training_set[h_i, :] - high_resolution_training_set[l_i, :])
#             x = a * np.square((sum(np.dot(W, combine)) - b * d))
#         else:
#             a = landa
#             b = landa/a
#             d = np.linalg.norm(high_resolution_training_set[h_i, :] - high_resolution_training_set[l_i, :])
#             x = a * np.square((sum(np.dot(W, combine)) - b * d))
#             indexyy = indexyy + 1
#         J_W = J_W + x



def A(h, l, h_labels, l_labels, landa):
    aa = 0
    index = 0
    for l_i, l_img in enumerate(l):
        for h_i, h_img in enumerate(h):
            combine_t = np.array([np.hstack((l_img, -h_img))])
            if h_labels[h_i] == l_labels[l_i]:
                a = (1 - landa) + landa
                the_a = a * combine_t.transpose().dot(combine_t)
            else:
                a = landa
                the_a = a * combine_t.transpose().dot(combine_t)
            aa = aa + the_a
            print(index + 1)
    return aa

def C(h, l, V, landa):
    cc = 0
    index = 0
    for l_i, l_img in enumerate(l):
        for h_i, h_img in enumerate(h):
            combine_t = np.array([np.hstack((l_img, -h_img))])
            q = sum(np.dot(V, combine_t.transpose()))
            d = np.linalg.norm(h[l_i, :] - h[h_i, :])
            if q > 0:
                the_c = landa * d / q * combine_t.transpose().dot(combine_t)
            else:
                the_c = 0
            cc = cc + the_c
    return cc

result = (A(high_resolution_training_set,low_resolution_training_set, high_training_labels, low_training_labels, landa))

#iterative majorization algorithm
t = 0
V = W.transpose()
a = A()
c = C()
a_1 = np.linalg.pinv(a)
W = np.dot(a_1, c).dot(V)

# #test
# print(images[0])
# img = cv2.imread(images[2], 0)
# cv2.imshow('image', img)
# k = cv2.waitKey(0)



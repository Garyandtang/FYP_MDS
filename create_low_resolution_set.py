import csv
import cv2
import os
# define a fixed list images = []*400
#if you want to use index: for i, item in enumerate(reader):

#get the image from csv file
csvFile = open("high_at.csv", "r")
reader = csv.reader(csvFile)
images = []
labels = []
for item in reader:
    images.append(item[0])
    labels.append(item[1])
csvFile.close()

if os.path.isdir("/home/jiawei/PycharmProjects/FYP_MDS/data/low_resolution") == True:
    print("directory is created")
else:
    os.mkdir("/home/jiawei/PycharmProjects/FYP_MDS/data/low_resolution")

# for a in range(40):
#     if a <= 9:
#         dirName = "/home/jiawei/PycharmProjects/FYP_MDS/data/low_resolution/s0%d" % (a + 1)
#     else:
#         dirName = "/home/jiawei/PycharmProjects/FYP_MDS/data/low_resolution/s%d" % (a + 1)
#     os.mkdir(dirName)

for item in images:
    image_path = item
    img = cv2.imread(image_path, 0)
    dest_path = "/home/jiawei/PycharmProjects/FYP_MDS/data/low_resolution/s" + image_path[53:55]
    low_resolution_name = dest_path + "/" + image_path[56:]
    #resize image to low resolution

    print(low_resolution_name)

    cv2.imwrite(low_resolution_name, img)
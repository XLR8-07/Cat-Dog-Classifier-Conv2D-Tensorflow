import numpy as np
import cv2
import os
import random 
import matplotlib.pyplot as plt
import pickle

TRAIN_DIRECTORY = r"E:\P R O D I G Y\A C A D E M I C\C O D E\DEEP LEARNING\Cat_Dog Classifier\dogscats\dogscats\train"
CATEGORIES = ['cats', 'dogs']
IMG_HEIGHT = 100
IMG_WIDTH = 100
data = []

for category in CATEGORIES:
    folder = os.path.join(TRAIN_DIRECTORY,category)
    # print(CATEGORIES.index(category))                             # Labels => 0 --> cat, 1 --> dog
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path  = os.path.join(folder,img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr,(IMG_HEIGHT,IMG_WIDTH))
        data.append([img_arr, label])
        # plt.imshow(img_arr)
        # plt.show()
        # break

# print(len(data))                                                  # 23K images in total
random.shuffle(data)

# Sorting out the featuers and labels
X = []      # features
y = []      # labels

for features, labels in data : 
    X.append(features)
    y.append(labels)

X = np.array(X)
y = np.array(y)

# Saving the features and labels for the whole dataset

pickle.dump(X , open('features.pkl','wb'))
pickle.dump(y , open('labels.pkl','wb'))

print(len(X))
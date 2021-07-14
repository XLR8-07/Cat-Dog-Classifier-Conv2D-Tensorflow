import pickle
import tensorflow as tf
import cv2
import numpy as np
import os


IMG_HEIGHT = 100
IMG_WIDTH = 100

folder = 'abedeen'
samples = []
results = []

model = tf.keras.models.load_model('Dogs&CatsClassifier2.h5')
# print(model.summary())

for img in os.listdir(folder):
    # print(img)
    img_path  = os.path.join(folder,img)
    img_arr = cv2.imread(img_path)
    img_arr = cv2.resize(img_arr,(IMG_HEIGHT,IMG_WIDTH))
    samples.append(img_arr)

X = np.array(samples)

predictions = model.predict(X)
result_arr =np.argmax(predictions, axis = 1)
print(result_arr)

for val in result_arr:
    if (val == 0):
        results.append('cat')
    else :
        results.append('dog')

print(results)
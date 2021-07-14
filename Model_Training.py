import pickle
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras import callbacks

NAME = f'cat-vs-dog-prediction-{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')

X = pickle.load(open('features.pkl','rb'))
y = pickle.load(open('labels.pkl','rb'))

X = X/255

# print(X.shape)

model = Sequential()

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128,input_shape = X.shape[1:], activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(2,activation='softmax'))  # For output, Sigmoid and Softmax works better , and there are 2 neurons in this layer ; Output layer is a hidden (dense) layer

model.compile(optimizer= 'adam' , loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

with tf.device('/gpu:0'):
    model.fit(X,y, epochs=20 , validation_split=0.2 , callbacks = [tensorboard])

model.save('Dogs&CatsClassifier2.h5')
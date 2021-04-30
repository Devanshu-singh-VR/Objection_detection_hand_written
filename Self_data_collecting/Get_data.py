import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import cv2 as cv
import pandas as pd

(train, label), (test, label_test) = tf.keras.datasets.mnist.load_data()
length = train.shape[0]

image_path = np.zeros(length).astype('str')
label = label
y_coord = np.zeros(length)
x_coord = np.zeros(length)
width = np.zeros(length)
height = np.zeros(length)

for i in range(length):
    black = np.zeros((75, 75))
    x = np.random.randint(0, 47)
    y = np.random.randint(0, 47)
    black[y:y+28, x:x+28] = train[i,:]

    cv.imwrite('train_images\\' + str(i) + '.png', black)

    image_path[i] = str(i)+'.png'
    x_coord[i] = x
    y_coord[i] = y
    width[i] = x+28
    height[i] = y+28
    print(i)

data = {"image_path": image_path, 'label': label, 'x_mark': x_coord, "y_mark": y_coord, 'width': width, "height": height}

frame = pd.DataFrame(data)
frame.to_csv('PATH', index=False) # Here you have to define the path
                                  # where you have to store the images dataset


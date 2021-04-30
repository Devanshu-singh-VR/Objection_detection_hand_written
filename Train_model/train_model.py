import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd

data = pd.read_csv('train_label.csv') # CSV file path

image = data.iloc[:, 0].values
label = data.iloc[:, 1:].values

train = tf.data.Dataset.from_tensor_slices((image, label))

def collector(images_file, label):
    image = tf.io.read_file('train\\'+images_file) # train_images file path
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)

    labels = {'label': label[0], 'coordinates': label[1:]}
    return image, labels

train = (
    train.shuffle(buffer_size=label.shape[0])
    .map(collector)
    .batch(batch_size=100)
)

# Using the functional API

rg = tf.keras.regularizers.l1(0.001)
input = tf.keras.Input(shape=(75, 75, 1))

x = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', kernel_regularizer=rg)(input)
x = tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.9)(x)

x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', kernel_regularizer=rg)(x)
x = tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.9)(x)

x = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', kernel_regularizer=rg)(x)
x = tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.9)(x)

x = tf.keras.layers.Flatten()(x)

# TWO output layers, one for label training and second for bounding box prediction

output1 = tf.keras.layers.Dense(10, activation='softmax', name="label")(x)
output2 = tf.keras.layers.Dense(4, name="coordinates")(x)

model = tf.keras.Model(inputs=input, outputs=[output1, output2])

# Two loss function 

model.compile(loss={"label": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),"coordinates": 'mean_squared_error'},
              optimizer='adam', metrics=['accuracy'])

model.fit(train, epochs=10, verbose=1)


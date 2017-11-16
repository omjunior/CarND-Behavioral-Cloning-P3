import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


#params
DELTA = 0.2
EPOCHS = 5

print("Reading CSV file")

lines = []
with open('./data/driving_log.csv', 'r') as cvsfile:
    reader = csv.reader(cvsfile)
    for line in reader:
        lines.append(line)

print("Reading image files")

images = []
measurements = []
for line in lines:
    path_c = './data/IMG/' + line[0].split('/')[-1]
    path_l = './data/IMG/' + line[1].split('/')[-1]
    path_r = './data/IMG/' + line[2].split('/')[-1]
    # remember that cv2 reads in BGR not RGB
    images.append(cv2.imread(path_c))
    images.append(cv2.imread(path_l))
    images.append(cv2.imread(path_r))
    measurements.append(float(line[3]))
    measurements.append(float(line[3])+DELTA)
    measurements.append(float(line[3])-DELTA)

print("Augmenting dataset")
aug_images, aug_measurements = [], []
for i, m in zip(images, measurements):
    aug_images.append(i)
    aug_measurements.append(m)
    aug_images.append(cv2.flip(i,1))
    aug_measurements.append(-1.0 * m)
images, measurements = None, None

print("Creating numpy arrays")

X_train = np.array(aug_images)
y_train = np.array(aug_measurements)

print("Building keras model")

from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mse', optimizer='Adam')

print("Training model")

history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=EPOCHS)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

print("Saving model")

model.save('model.h5')

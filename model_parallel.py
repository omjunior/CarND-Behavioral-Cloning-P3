import csv
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.model_selection import train_test_split

# hyperparams
DELTA = 0.2
EPOCHS = 5
BATCH = 40

print("Reading CSV file")
lines = []
with open('./data/driving_log.csv', 'r') as cvsfile:
    reader = csv.reader(cvsfile)
    next(reader, None) #skip header
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def process_sample(sample):
    # 0 = center, 1 = left, 2 = right
    path_c = './data/IMG/' + sample[0].split('/')[-1]
    path_l = './data/IMG/' + sample[1].split('/')[-1]
    path_r = './data/IMG/' + sample[2].split('/')[-1]
    image_c = cv2.cvtColor(cv2.imread(path_c), cv2.COLOR_BGR2RGB)
    image_l = cv2.cvtColor(cv2.imread(path_l), cv2.COLOR_BGR2RGB)
    image_r = cv2.cvtColor(cv2.imread(path_r), cv2.COLOR_BGR2RGB)
    steering = float(line[3])
    flip = np.random.random()
    if (flip > 0.5):
        image_c = cv2.flip(image_c, 1)
        image_l = cv2.flip(image_r, 1)
        image_r = cv2.flip(image_l, 1)
        steering = -1.0 * steering
    return image_c, image_l, image_r, steering

def process_sample_no_aug(sample):
    path_c = './data/IMG/' + sample[0].split('/')[-1]
    path_l = './data/IMG/' + sample[1].split('/')[-1]
    path_r = './data/IMG/' + sample[2].split('/')[-1]
    image_c = cv2.cvtColor(cv2.imread(path_c), cv2.COLOR_BGR2RGB)
    image_l = cv2.cvtColor(cv2.imread(path_l), cv2.COLOR_BGR2RGB)
    image_r = cv2.cvtColor(cv2.imread(path_r), cv2.COLOR_BGR2RGB)
    steering = float(line[3])
    return image_c, image_l, image_r, steering

def generator(samples, batch_size, augment):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images_c = []
            images_l = []
            images_r = []
            angles = []
            for batch_sample in batch_samples:
                if (augment):
                    imc, iml, imr, an = process_sample(batch_sample)
                else:
                    imc, iml, imr, an = process_sample_no_aug(batch_sample)
                images_c.append(imc)
                images_l.append(iml)
                images_r.append(imr)
                angles.append(an)
            X_train = [np.array(images_c), np.array(images_l), np.array(images_r)]
            y_train = np.array(angles)
            yield X_train, y_train

train_generator = generator(train_samples, BATCH, True)
validation_generator = generator(validation_samples, BATCH, False)


print("  Train set with {} samples".format(len(train_samples)))
print("  Validation set with {} samples".format(len(validation_samples)))

print("Building keras model")
from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.callbacks import EarlyStopping, TensorBoard
from keras.engine.topology import Merge

# based on:
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
model_c = Sequential()
model_c.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
model_c.add(Lambda(lambda x: x/255.0 - 0.5))
model_c.add(Convolution2D(24, 5, 5, activation="relu", subsample=(2, 2)))
model_c.add(Convolution2D(36, 5, 5, activation="relu", subsample=(2, 2)))
model_c.add(Convolution2D(48, 5, 5, activation="relu", subsample=(2, 2)))
model_c.add(Convolution2D(64, 3, 3, activation="relu"))
model_c.add(Convolution2D(64, 3, 3, activation="relu"))
model_c.add(Flatten())
model_c.add(Dense(100, activation="relu"))

model_l = Sequential()
model_l.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
model_l.add(Lambda(lambda x: x/255.0 - 0.5))
model_l.add(Convolution2D(24, 5, 5, activation="relu", subsample=(2, 2)))
model_l.add(Convolution2D(36, 5, 5, activation="relu", subsample=(2, 2)))
model_l.add(Convolution2D(48, 5, 5, activation="relu", subsample=(2, 2)))
model_l.add(Convolution2D(64, 3, 3, activation="relu"))
model_l.add(Convolution2D(64, 3, 3, activation="relu"))
model_l.add(Flatten())
model_l.add(Dense(100, activation="relu"))

model_r = Sequential()
model_r.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
model_r.add(Lambda(lambda x: x/255.0 - 0.5))
model_r.add(Convolution2D(24, 5, 5, activation="relu", subsample=(2, 2)))
model_r.add(Convolution2D(36, 5, 5, activation="relu", subsample=(2, 2)))
model_r.add(Convolution2D(48, 5, 5, activation="relu", subsample=(2, 2)))
model_r.add(Convolution2D(64, 3, 3, activation="relu"))
model_r.add(Convolution2D(64, 3, 3, activation="relu"))
model_r.add(Flatten())
model_r.add(Dense(100, activation="relu"))

model = Sequential()
model.add(Merge([model_c, model_l, model_r], mode='concat'))
model.add(Dense(50, activation="relu"))
model.add(Dense(1))

model.compile(loss='mse', optimizer='Adam')

print("Training model")
# earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, \
#                           verbose=1, mode='auto')
# tensorboard = TensorBoard(log_dir='./logs')
history = model.fit_generator(generator=train_generator, \
    samples_per_epoch=len(train_samples), \
    validation_data=validation_generator, \
    nb_val_samples=len(validation_samples), \
    nb_epoch=EPOCHS, \
    callbacks=[])

print("Saving model")
model.save('model_parallel.h5')

# plot losses
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.semilogy(10)
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

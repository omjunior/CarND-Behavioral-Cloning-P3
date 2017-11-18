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
    choice = np.random.choice([0, 1, 2])
    path = './data/IMG/' + sample[choice].split('/')[-1]
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    steering = float(line[3]) * (((choice + 1) % 3) - 1) * DELTA
    flip = np.random.random()
    if (flip > 0.5):
        image = cv2.flip(image, 1)
        steering = -1.0 * steering
    return image, steering

def process_sample_no_aug(sample):
    path = './data/IMG/' + sample[0].split('/')[-1]
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    steering = float(line[3])
    return image, steering

def generator(samples, batch_size, augment):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                if (augment):
                    im, an = process_sample(batch_sample)
                else:
                    im, an = process_sample_no_aug(batch_sample)
                images.append(im)
                angles.append(an)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, BATCH, True)
validation_generator = generator(validation_samples, BATCH, False)

print("  Train set with {} samples".format(len(train_samples)))
print("  Validation set with {} samples".format(len(validation_samples)))

print("Building keras model")
from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.callbacks import EarlyStopping, TensorBoard

# based on:
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
model = Sequential()
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
# 75 x 320 x 3
model.add(Convolution2D(24, 5, 5, activation="relu", subsample=(2, 2)))
# 36 x 158 x 24
model.add(Convolution2D(36, 5, 5, activation="relu", subsample=(2, 2)))
# 16 x 77 x 36
model.add(Convolution2D(48, 5, 5, activation="relu", subsample=(2, 2)))
# 6 x 37 x 48
model.add(Convolution2D(64, 3, 3, activation="relu"))
# 4 x 35 x 64
model.add(Convolution2D(64, 3, 3, activation="relu"))
# 2 x 33 x 64
model.add(Flatten())
# 4224
model.add(Dense(100, activation="relu"))
# 100
model.add(Dense(50, activation="relu"))
# 50
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
model.save('model.h5')

# plot losses
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.semilogy(10)
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

import csv
import cv2
import sklearn
import math
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.model_selection import train_test_split

# hyperparams
DELTA_CAMERA = 0.25
DELTA_SHIFT = 0.005
EPOCHS = 5
BATCH = 40

def steering_histogram(all_data):
    steering = []
    for line in all_data:
        steering.append(float(line[3]))
    st_np = np.array(steering)
    hist = np.histogram(np.abs(st_np), bins=10, range=(0, 1))
    hist = hist[0] / np.sum(hist[0])
    return hist

def reduce_steering_bias(all_data, hist):
    new_data = []
    for i in range(0, 3):
        for line in all_data:
            steering = float(line[3])
            dup = np.random.random()
            if (dup < ((1 - hist[min(9, int(math.floor(abs(steering)/0.1)))]) ** 2) ):
                new_data.append(line)
    all_data.extend(new_data)
    return all_data

def process_sample(sample):
    # select one camera
    # 0 = center, 1 = left, 2 = right
    choice = np.random.choice([0, 1, 2])
    path = './data/IMG/' + sample[choice].split('/')[-1]
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    steering = float(sample[3]) + (((choice + 1) % 3) - 1) * DELTA_CAMERA
    # flips the image
    flip = np.random.random()
    if (flip > 0.5):
        image = cv2.flip(image, 1)
        steering = -1.0 * steering
    # brightness
    bounded = True
    bright = np.random.normal(loc=1, scale=0.3)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = np.uint8(np.minimum(255, hsv[:,:,2] * bright))
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # shifting
    shift_h = np.random.normal(loc=0, scale=10)
    shift_v = np.random.normal(loc=0, scale=10)
    steering = steering + shift_h * DELTA_SHIFT
    M = np.float32([[1, 0, shift_h], [0, 1, shift_v]])
    image = cv2.warpAffine(image, M, (320, 160))

    return image, steering

def process_sample_no_aug(sample):
    path = './data/IMG/' + sample[0].split('/')[-1]
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    steering = float(sample[3])
    return image, steering

def show_image_examples(samples):
    n = len(samples)
    orig = []
    for i in range(0, n):
        orig.append(process_sample_no_aug(samples[i]))
    proc = []
    for i in range(0, n):
        proc.append(process_sample(samples[i]))

    plt.figure()
    for s in range(0, n):
        plt.subplot(2, n, s+1)
        plt.title(orig[s][1])
        plt.axis('off')
        plt.imshow(orig[s][0])
    for s in range(0, n):
        plt.subplot(2, n, n+s+1)
        plt.title(proc[s][1])
        plt.axis('off')
        plt.imshow(proc[s][0])
    plt.show()

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


print("Reading CSV file")
lines = []
with open('./data/driving_log.csv', 'r') as cvsfile:
    reader = csv.reader(cvsfile)
    next(reader, None) #skip header
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# print("Before")
percent_hist = steering_histogram(train_samples)
# print(percent_hist)
train_samples = reduce_steering_bias(train_samples, percent_hist)
# print("After")
# percent_hist = steering_histogram(train_samples)
# print(percent_hist)

# shuffle(train_samples)
# show_image_examples(train_samples[0:5])

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

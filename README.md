# Behavioral Cloning Project

[//]: #
[modelimage]: ./model.png "Model Visualization"
[augmentimage]: ./augment.png "Images Augmentation"

## Objectives

The main goal of this project is to use captured data from human driving of a
given simulator to train a model that should be able to drive autonomously the
same simulator.

## Rubric Points

### Required Files
The submission contains this writeup (```README.md```), a python file that
defines and trains the model (```model.py```), a saved trained model file
(```model.h5```) and a python file that uses the trained model to drive the
simulation autonomously.

### Code
The code can easily be used to train a model by running the following code,
ensuring that a suitable data folder exists on the same directory.
```
python model.py
```
This will generate a ```model.h5``` file in the same directory.

To run the model, the following line must be run, along with the simulator.
```
python drive.py model.h5
```

The images are fed to the Keras model by means of a python generator.
There are two generators defined, one for the validation data, which just
generate a batch of images coming from the central camera.

The other generator is used to train the model. For each data point, random
augmentations are applied and a batch of these augmented images are fed to
the model.

### Model Architecture and Training Strategy

#### Data
Rather than recording data from the simulator myself, I preferred to use only
the data provided by Udacity. Besides using a computer with very poor graphics
performance, I also don't have available an analog input, such as a joystick, to
provide good data. Since my data was limited, I have resorted to data
augmentation to sufficiently train the model.

The first thing to notice is that there are much more training points at low
steering angles. So in order to reduce this bias, I have made controlled data
duplication.

I have made a histogram of the absolute value of steering and computed their
percentages. After that, for every line encountered I have duplicated the line
with the probability equals to (1 - histogram_percentage) squared. By running
this procedure three times, the portion of steering angles between 0 and 0.1
dropped from approx. 74% to approx. 50% while roughly doubling the input lines.

After that, the images produced by the generator suffer a series of random
transformations, as follows:

* One of the three cameras is selected at random. Empirically it could be seen
that selecting more center images produces a better result. So the probabilities
chosen were 50% for the center image and 25% for each other camera (left or
right). In the case a lateral camera was chosen, the value of 0.20 was added or
subtracted, to simulate a steering that would get the car back to the track
center.

* A horizontal flip is performed and the steering angle changes signal, with
50% probability.

* Brightness augmentation: The color space is transformed from RGB to HSV, and
the V channel gets multiplied by a normal random (mu=0, sigma=0.3). Then the
image is brought back to RGB color space.

* Translation: The image is translated horizontally and vertically by a random
normal amount (mu=0, sigma=10). For each pixel shifted horizontally, a
correction of 0.005 is performed on the steering angle.

This way, at each epoch, slightly different images are fed to the model, almost
as if more data was acquired. Follow examples below:

![alt-text][augmentimage]

#### Model

The model was strongly based on the NVIDIA network architecture found
[here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

Follow a visualization of the used model.

![alt text][modelimage]

It first crops the image from its original size (160, 320, 3) to (75, 320, 3) by
removing the top 60 and the bottom 25 pixel rows, which consist mostly of
background and hood of the car.

After that, a normalization and zero mean is performed. In the sequence, there
are three 5x5 and two 3x3 convolutional layers followed by three fully connected
layers. Some 50% dropout layers have been added to reduce overfitting, which
has improved a lot the quality of the trained model experimentally.

All activation functions were defined as ReLU functions. The model used an Adam
optimizer, which according to [Sebastian Rudder](http://ruder.io/optimizing-gradient-descent/)
is the overall choice to be used.

The loss function used was the minimum square error, which is adequate for when
the model makes a numerical prediction rather than a classification.

An embedded early stopping functionality from Keras was used, so the model will
stop as soon as the validation loss stops diminishing, further reducing
overfitting.

I have set up to mini-batch sizes of 512 samples, and 20 epochs - although
early stopping kicked in after 11 epochs.

#### Driving

I have made a slight change on the ```drive.py``` script. The target speed is
now a function of the steering angle applied. If the car is going straight the
target speed is 30 mph, if it the steering angle is maximum the target speed
is 15 mph. Anywhere in between a linear function is applied.

### Conclusion

Even with a limited amount of data, the model was able to perform well on the
track it was trained, as can be seen on the [video recording](./video.mp4).
In some parts of the track it 'snakes' a little bit, maybe because of
overcorrecting the steering when selecting side cameras or when shifting the
training images, but I couldn't reduce this behavior any further. Possibly more
data could reduce this unwanted behavior.

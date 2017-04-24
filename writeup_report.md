#**Behavioral Cloning** 

##Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/cropped_center_2017_04_20_09_39_56_653.jpg "Cropped Image"
[image2]: ./images/center_2017_04_20_09_39_15_969.jpg "Center Image"
[image3]: ./images/center_2017_04_20_09_39_49_881.jpg "Center Image 2"
[image4]: ./images/left_2017_04_20_09_39_49_881.jpg "Left Image"
[image5]: ./images/right_2017_04_20_09_39_49_881.jpg "Right Image"
[image6]: ./images/center_2017_04_20_09_40_00_020.jpg "Normal Image"
[image7]: ./images/flipped_center_2017_04_20_09_40_00_020.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 containing a video recording of the vehicle driving autonomously for one lap around the track

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I decide to use the NVIDIA architecture for my model and consists of three convolutional layers with 5x5 filter sizes and depths between 3 (RGB image) and 36 (model.py lines 83-90), two more convolutional layers with 3x3 filter sizes and depths between 32 and 64 (model.py lines 92-96), a flatten layer and four fully-connected layers (model.py lines 98-111).

The model includes RELU layers on the convolutional layers to introduce nonlinearity (code line 83-96), and the data is normalized in the model using a Keras lambda layer (code line 78). 

####2. Attempts to reduce overfitting in the model

I decided not to use dropout layers due to the low difference between training loss and validation loss of the model at each epoch.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 13). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving for six laps (three laps in a clock-wise direction and another three laps in a counter-clockwise direction for better generalization), recovering from the left and right sides of the road and i also used the right and left cameras images applying a correction on the steering angle for help the vehicle to recover from the sides of the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to trying different model architectures to see how the vehicle behaves.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it's a powerful architecture to work with images. I noticed that the training and validation loss wasn't low enough. So a tried another architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had a high mean squared error on the training set and a high mean squared error on the validation set. This implied that the model was underfitting. 

To combat the underfitting, I used the NVIDIA architecture because it's been used for the same purpose and it has demonstrated good peformance in simulation and on-road tests.

Then I achieved a low mean squared error for both training and validation set.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. The biggest problem was when the car deviated from the center line, it wouldn't recover and end up fell off the track. To improve the driving behavior in these cases, I decided to use the left and right images from the car and used a steering correction of 0.3 (model.py lines 33-35).

I did not noticed any sign of overfitting so I decided not to use dropout layers on the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 77-110) consisted of a convolution neural network with the following layers and layer sizes:

* Normalization layer (line 77)
* Cropping layer (line 80)
* Three convolutional layers with a 2x2 strides, 5x5 kernel and ReLU activations
* Two non-strided convolutional layers with a 3X3 kernel and ReLu activations
* Flatten layer
* Three fully-connected layers
* Output layer providing the steering angle

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded six laps (three for each direction) on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I also used the left and right images generated by the other two cameras of the car to simulate the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to stay at the center of the road. These images show the center, left and right images at a specific moment:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would insert more data and generalize the data set. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I also converted all the images to RGB.

After the collection process, I had 40530 (20265 center, left and right images) number of data points. I then preprocessed this data by normalizing and mean centered the data. After that, I cropped the images by trimming 70 pixels from the top and 25 pixels from the bottom. Here is an image after the cropping layer:

![alt text][image1]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the training loss that wasn't getting lower. I used an adam optimizer so that manually training the learning rate wasn't necessary.
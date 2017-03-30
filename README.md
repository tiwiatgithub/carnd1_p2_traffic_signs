## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
The goal of this project, is to use deep neural networks and convolutional neural networks to classify traffic signs. The data to train and validate the model will be the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

After training the model, the model is tested on German traffic signs that can be found on the web.

The project contains three major files:
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report

The model is inspired by  [Inception-ResNet-v1](https://arxiv.org/abs/1602.07261). Although it is build with fewer layers, it includes on implementation of each of the three inception blocks A,B and C.

Training the model with the entire training data set for 20 epochs and a batch size of 128 takes approximately 25 minutes on an _AWS E2C g2.2xlarge_ instance and reaches a performance of around 94 % on the validation set. There is however room for improvement.


The Project
---
The steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Preprocess the data
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This project requires:
* Python 3
* Tensorflow
* numpy
* pandas
* matplotlib
* sklearn
* zipfile
* PIL
* pickle
* os
* cv2

### Run the model
In order to run the model the dataset [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) has to be downloaded and the path within the jupyther notebook has to be set accordingly. The easiest way the cover the dependencies is by intalling [Anaconda](https://www.continuum.io/downloads) and follwing the installation instruction for the given operating system.
By now tensorflow is not included within Anaconda, but the installtion is also well documented and can be found [here](https://www.tensorflow.org/install/).

Having everything set up run the juypter notebook and make adjustments where desired.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

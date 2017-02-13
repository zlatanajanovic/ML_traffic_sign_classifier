#**Traffic Sign Recognition** 

##Writeup



---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/raw.png "Training set raw images"
[image2]: ./examples/count.png "Histogram of traffic sign distribution"
[image3]: ./examples/equalized.png "Histogram equalized images"
[image4]: ./test/1big.jpg "Traffic Sign 1"
[image5]: ./test/2big.jpg "Traffic Sign 2"
[image6]: ./test/3big.jpg "Traffic Sign 3"
[image7]: ./test/4big.jpg "Traffic Sign 4"
[image8]: ./test/5big.jpg "Traffic Sign 5"
[image9]: ./test/6big.jpg "Traffic Sign 6"
[image10]: ./examples/training_LeNet.png "Training on LeNet lab model"
[image11]: ./examples/training.png "Training on final model"
[image12]: ./examples/test_images.png "Test images adjusted and preprocessed"
[image13]: ./examples/test_images1.png "Test images classified"
[image14]: ./examples/softmax.png "Softmax values of model prediction"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/zlatanajanovic/ML_traffic_sign_classifier)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed in training set.

![alt text][image2]

Some random examples of a training set look like this.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to equailize histogram of images, because some images were much darker than others.

Here is an example of a traffic sign images before and after histogram equalization.

![alt text][image1]

![alt text][image3]

As a last step, I normalized the image data because in training model gets better accuracy faster.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook.  

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x6 	|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x12 				|
| Dropout				| Keep prob conv								|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 14x14x24 	|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 12x12x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x48 					|
| Dropout				| Keep prob conv								|
| Fully connected		| 1728 -> 512									|
| RELU					|												|
| Dropout				| Keep prob										|
| Fully connected		| 512 -> 128									|
| RELU					|												|
| Dropout				| Keep prob										|
| Fully connected		| 128 -> 43										|
| Dropout				| Keep prob										|
| Softmax				| 	 											|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth, nineth and tenth cell of the ipython notebook. 

To train the model, I used an Adam optimizer with learning rate = 0.0005. The learning is done in 50 epochs with batch size of 64. For radom initialization of weights mu 0 and sigma 0.1 values were used, and bias was initialized to zero.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eight cell of the Ipython notebook. It is called after each epoch of training in the ninth cell.

The process of obtaining adequate model was iterative, mixed with debugging and parameter tuning. Start was with LeNet architecture from LeNet lab for handwriting digits classification. As shown on a image bellow this model was overfitting. For this reason dropout was used and model was scaled up a bit. 

![alt text][image10]

Training of final model can be seen bellow.

![alt text][image11]


My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.932
* test set accuracy of 0.920

During training batch size was adjusted, and it was shown that big batch size leads to underfitting.
Two convolutional layers per one max pooling were used to get bigger model complexity. The convolution kernel was decrased to 3x3 as it showed to give better results.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image8] 
![alt text][image9] ![alt text][image7] 

After resizing, cropping and preprocessing images look like this

![alt text][image12]
The second and the last image might be difficult to classify because they are not centered and complete on the image. And "Share the road" sign is not present in dataset at all.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the fourteenth cell of the Ipython notebook.

Here are the results of the prediction:
![alt text][image13]

| Image					|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way	 		| Right-of-way									| 
| Priority road			| Priority road									|
| No entry				| No entry										|
| Children crossing		| Children crossing								|
| Share the road		| 30 km/h										|
| Stop Sign      		| Stop sign										| 


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 83.33%. This compares favorably to the accuracy on the test set of 92 %

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making analysis on my final model is located in the 15th cell of the Ipython notebook.

For all images which were classified correctly model was very certain (~100%). All softmax probabilities can be seen on picture.

![alt text][image14]

For the "Share the road" image, the model was not sure what it is. All probabilities were low, with "30 km/h" probabliliti highest (probability of 6%). The reason for this is that this sign doesn't exist in dataset. The top five soft max probabilities were:


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .06         			| 30 km/h   									| 
| .06     				| Right-of-way									|
| .05					| Ahead only									|
| .04	      			| 80 km/h						 				|
| .04				    | Priority Road      							|


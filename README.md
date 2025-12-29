## 🌊💧🔥 Tensor Flow 🔥💧🌊

- Of course, machine learning always needs data. In the initial stage we need to understand the dataset we have first. Some things you need to know are the format of the data, number of samples, and how many labels. Apart from that, we also need to ensure whether the dataset is continuous data (regression problem) or discrete data (classification problem).
- Unseen Data Evaluation The model is exposed to entirely new images that were neither part of the training nor the validation sets. This ensures the model has truly learned to recognize "messiness" rather than just memorizing the specific pixels of the training images.
- Preprocessing Consistency Test images must undergo the exact same resizing and normalization steps used during the training phase. If the model was trained on $150 \times 150$ pixel images, a test image of a different size must be reshaped to match those dimensions before a prediction can be made.

The dataset we use has 192 training data samples consisting of 96 samples of neat room images and 96 samples of messy room images.
The stages of this training are:

Ensure TensorFlow used in Google Colab is version above 2.0.
1. Download the dataset and extract the file using the unzip method.
2. Place the directory for each class in the train and validation directories into variables.
3. Pre-processing data with image augmentation.
4. Prepare the training data that the model will learn from.
5. Build a model architecture with a Convolutional Neural Network (CNN).
6. Compile and train the model with model.compile and model.fit until you get the desired accuracy.
7. Test the model that has been created using images that are not yet recognized by the model.

## 🌊💧🔥 Tensor Flow Version 🔥💧🌊
The first thing you need to do is make sure that the version of TensorFlow you are using is version 2 or above.
- import tensorflow as tf
- print(tf.__version__)

## 🌊💧🔥 Prepare Dataset 🔥💧🌊
The next stage is preparing the dataset that will be used. The code below functions to extract the data that we previously downloaded. Then we define directory names for training data and validation data.

```
import zipfile, os
local_zip = '/tmp/messy_vs_clean_room.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
 
base_dir = '/tmp/images'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
```

## 🌊💧🔥 Finding Sub Directory 🔥💧🌊
After you run the code above, pay attention, the training data and validation data directories each have clean and messy sub-directories. Each sub-directory stores images corresponding to that sub-directory name. So, in the 'clean' sub-directory there are pictures of neat rooms and in the 'messy' sub-directory there are pictures of messy rooms.

```
os.listdir('/tmp/images/train')
os.listdir('/tmp/images/val')
```

![image](https://github.com/diantyapitaloka/Tensor-Flow/assets/147487436/46022987-b029-45fc-b6d5-c132eb87721c)


## 🌊💧🔥 Image Data Generator 🔥💧🌊
In the next step, we will apply ImageDataGenerator to training data and validation data. ImageDataGenerator is a very useful function for preparing training data and validation data. Some of the conveniences provided by ImageDataGenerator include data preprocessing, automatic sample labeling, and image augmentation.

Image augmentation is a technique that can be used to increase training data by duplicating existing images by adding certain variations. 

![image](https://github.com/diantyapitaloka/Tensor-Flow/assets/147487436/78c6137a-028c-4d45-92ff-cd6b5816d930)


## 🌊💧🔥 Image Augmentation Process 🔥💧🌊
The following code shows the image augmentation process for each sample in the dataset.
```
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
 
![image](https://github.com/diantyapitaloka/Tensor-Flow/assets/147487436/18273ad1-9124-4061-89eb-bfdf4e7c09dd)


## 🌊💧🔥 Convolutional Neural Network Model 🔥💧🌊
Once the data is ready, we can build a Convolutional Neural Network (CNN) model. Creating a CNN model in Keras is similar to creating a Multi Layer Perceptron (MLP) model discussed in the previous module. The difference is in the four layers of convolution layers and max pooling.

In the CNN model, the image classification process only focuses on the unique attributes that differentiate each category. So, this technique is considered more optimal than just using the MLP model which differentiates each category by looking at all the pixels in the image.

## 🌊💧🔥 Model Architecture Summary 🔥💧🌊
After creating the model, we can use the summary() function to see a summary of the model architecture that we have created.
```
model.summary()
```

![image](https://github.com/diantyapitaloka/Tensor-Flow/assets/147487436/a7482fc6-9cc5-428a-bd82-29d59b954625)

![image](https://github.com/diantyapitaloka/Tensor-Flow/assets/147487436/05e577bf-c313-4771-b94e-3c94623207d0)

## 🌊💧🔥 Compile Models 🔥💧🌊
Based on the summary results above, the model we created consists of four Convolutional layers and a MaxPoling layer, a flatten layer, and two dense layers. Remember that the last dense layer is the output layer. In the case of binary classification, the model output is a single number between 0 and 1. So, we set the last dense layer = 1. Meanwhile, the "Param #" column contains information about the number of parameters in each layer.

Next, the "Output Shape" column contains information on the size of the output produced by each layer. If you pay attention, the input image size that was previously defined is (150, 150). But in the first convolutional layer, each input image will produce an output size (148, 148) of 32 images. This size is reduced because we use filters with size (3, 3) with a total of 32 filters. So, each input image will produce 32 new images with size (148, 148).

Then, the resolution of each image will be reduced while maintaining the information in the image using a MaxPoling layer of size (2, 2). This will result in an output image size of (74, 74). Well, this process also applies to other Convolutional and MaxPoling layers.

Next, let's look at the flatten layer. The output from the last MaxPoling layer consisting of 512 images with size (7, 7) will be converted into a 1D array (1D tensor). This will produce an output of size (25088).

So, the output then goes into the first dense layer which has 512 neurons. So, it will produce an output of size (512). Next, this output will enter the second dense layer which has 1 neuron so that it will produce an output of size (1). The output from this last layer is used as the final model result for binary classification cases.

Compile model with 'adam' optimizer loss function
```
'binary_crossentropy'
model.compile(loss='binary_crossentropy',
optimizer=tf.optimizers.Adam(),
metrics=['accuracy'])
```

## 🌊💧🔥 Fitting Model 🔥💧🌊
So, the final stage of model making is a process called model fitting. It is a process for training a model on input data and corresponding labels. In this process, we enter training data into the Neural Network network that we created previously.

The things that must be defined at this stage are the loss function and optimizer. Then, we start the model training process by calling the fit() function.

By using ImageDataGenerator, we don't need to enter image parameters and labels. Image data generator automatically labels images according to their directory. For example, an image contained in the clean directory will be labeled "clean" by ImageDataGenerator automatically.

How many batches will be executed at each epoch
```
model.fit( train_generator, steps_per_epoch=25,
```

Add epochs if model accuracy is not optimal
```
epochs=20
```

Displays validation data testing accuracy
```
validation_data=validation_generator, 
```

How many batches will be executed at each epoch
```
validation_steps=5, verbose=2
```

## 🌊💧🔥 Final Visualization 🔥💧🌊 
The following is an example of displaying the prediction results from the previous model.
The final visualization as below :

![image](https://github.com/diantyapitaloka/Tensor-Flow/assets/147487436/0e08c9f8-35c4-498d-99d8-f050e82f8898)

## 🌊💧🔥 License 🔥💧🌊
- Copyright by Diantya Pitaloka

## 🌊💧🔥 Tensor Flow 🔥💧🌊

- Of course, machine learning always needs data. In the initial stage we need to understand the dataset we have first. Some things you need to know are the format of the data, number of samples, and how many labels. Apart from that, we also need to ensure whether the dataset is continuous data (regression problem) or discrete data (classification problem).
- Unseen Data Evaluation The model is exposed to entirely new images that were neither part of the training nor the validation sets. This ensures the model has truly learned to recognize "messiness" rather than just memorizing the specific pixels of the training images.
- Test the model using different camera hardware or in varied lighting conditions, such as nighttime or harsh sunlight. Hence, this is cause "stress test" reveals if your model is too really dependent on the high-quality lighting typical of curated datasets.
- Use heatmaps to visualize which pixels the model is focusing on when it makes a classification decision. Hence, this ensures the model is actually looking at the "mess" on the floor rather than just identifying a specific color of carpet or wallpaper.
- Hardware-Aware Architecture Search: If deploying to specific hardware like an Edge TPU, design your model using operations that are natively supported by that chip. Some complex custom layers might not be compatible with hardware acceleration, leading to slow "fallback" execution on the CPU.
- Cross-Validation for Statistical Power: Divide your data into multiple rotating folds to ensure most every sample is used for both training and testing across different iterations. This provides a more rigorous estimate of how the model will perform in the wild compared to a single static split.
- Anchor Box Optimization for Detection: Adjust the predefined bounding box scales to match the typical size and aspect ratio of the objects you expect to find. Fine-tuning these parameters ensures the localization system is primed to capture everything from tiny crumbs to large scattered items.
- Residual Connections for Deep Networks: Add skip connections that allow gradients to flow through the network without passing through every single non-linear activation. This architectural tweak prevents the "vanishing gradient" problem and enables the successful training of much deeper and more complex models.
- Synthetic Data Generation: Utilize 3D rendering engines to create artificial images of cluttered environments with like perfectly labeled ground truth data. This approach supplements your real-world images and provides the model with thousands of edge cases that would be difficult to photograph manually.
- Ensemble Predictions for Reliability: Instead of relying on a single model, run multiple versions of your architecture (trained with different seeds) and average their outputs. This "wisdom of the crowd" approach can smooth out individual model biases and provide a more stable confidence score for your smart-home app.
- Batch Normalization for Faster Convergence: Integrate Batch Normalization layers after your convolutional blocks. This technique re-centers and re-scales the inputs to each layer, which stabilizes the learning process and significantly reduces the number of epochs required to reach peak accuracy.
- Pruning for Inference Speed: Identify and remove redundant neural connections that contribute minimally to the final prediction accuracy after the training phase is complete. This process creates a "sparser" model that requires fewer computational cycles to process a single frame on local hardware.
- Hyperparameter Tuning with Grid Search: Systematically experiment with different optimizer types and momentum values to find the most efficient in path for weight updates. Identifying the ideal configuration can lead to a significant jump in accuracy without changing a single line of the core architecture.
- Class Imbalance Mitigation: Use oversampling or weighted loss functions if your dataset contains significantly more "clean" samples than "messy" ones. This prevents the model from developing a bias toward the majority class and ensures it remains sensitive to rare but important visual cues.
- Feature Map Visualization: Inspect the intermediate outputs of your convolutional layers to understand what geometric shapes or textures the model is detecting. This diagnostic step helps verify that the early stages of the network are capturing relevant structural details rather than random noise.
- Silent Mode Deployment (Shadow Testing): Before fully launching a new model version to users, run it in "shadow mode" where it makes predictions in the background without triggering notifications. Compare these silent predictions against the current live model to verify real-world improvements before a full rollout.
- Cross-Validation for Robustness: Instead of a single train/test split, use K-Fold Cross-Validation during the experimental phase. Hence, this involves rotating which portion of the data is used for validation across multiple runs, ensuring that your model's high performance isn't just a result of a "lucky" data split.
- Loss Function Selection: Evaluate if Binary Cross-Entropy is truly sufficient. If your "messy" vs "neat" labels have a lot of visual overlap, experimenting with Focal Loss can help the model focus more on "hard" examples (ambiguous rooms) that it frequently misclassifies, rather than getting "easy" examples right over and over.
- Hyperparameter Tuning with KerasTuner: Don't guess the number of filters or the dropout rate. Hence, use an automated tuning framework to systematically search through "search spaces" for the optimal number of neurons and layers, ensuring your architecture is mathematically optimized rather than just "good enough."
- Activation Map Analysis: Utilize Global Average Pooling (GAP) layers instead of dense flattening to reduce the total number of parameters. This not only discourages overfitting but also allows you to generate Class Activation Maps (CAM) more easily to see exactly which "messy" features are triggering the model.
- Synthetic Data Augmentation: If you lack photos of specific messy scenarios (like a spilled drink on a rug), use image synthesis or GANs (Generative Adversarial Networks) to create artificial training samples. This helps fill gaps in your dataset without requiring manual photography of every possible household mishap.
- The default 0.5 probability threshold isn't always ideal for every use case. You might decide a room is only "messy" if the confidence is above 0.8 to avoid annoying users with false positives in a smart-home app.
- If your initial testing shows low accuracy, experiment with adjusting the learning rate or adding Dropout layers to prevent overfitting. Finding the "sweet spot" in your architecture is often the difference between a prototype and a production-ready model.
- Check if the model struggles with rotated images or different brightness levels during testing. If it does, you should apply random flips, rotations, and zooms to your training set to help the model become invariant to camera angles.
- If you plan to run your model on a mobile device or a Raspberry Pi, use TensorFlow Lite to quantize the weights from float32 to int8. This significantly reduces the model size and speeds up processing without a massive loss in accuracy.
- If you plan to run your model on a mobile device or a Raspberry Pi, use TensorFlow Lite to quantize the weights from float32 to int8. This significantly reduces the model size and speeds up processing without a massive loss in accuracy.
- Beyond simple accuracy, you should generate a confusion matrix to see specifically which classes are being swapped. This helps identify if the model is biased toward "neat" rooms or if certain "messy" features are consistently confusing it.
- Preprocessing Consistency Test images must undergo the exact same resizing and normalization steps used during the training phase. If the model was trained on $150 \times 150$ pixel images, a test image of a different size must be reshaped to match those dimensions before a prediction can be made.
- Probability Interpretation When the model processes an image, it outputs a numerical value (often between 0 and 1) representing its confidence. A score close to 0 might represent a "neat" room, while a score close to 1 represents a "messy" room, based on how your labels were encoded.
- Inference Speed Testing allows you to measure how long it takes for the model to process a single image, known as inference time. This is a critical metric if you plan to deploy the model into a real-time app or a smart-home device.
- Generalization Audit By reviewing where the model fails—such as misclassifying a room because of unique lighting or shadows—you gain insights into how to improve the dataset. This feedback loop helps you decide if you need more diverse training samples to make the model more robust.
- Learning Rate Schedulers: Instead of using a fixed learning rate, implement a decay schedule that gradually reduces the step size as training progresses. This allows the model to settle into the global minimum of the loss function more effectively during the final epochs.
- Early Stopping Implementation: Monitor the validation loss during training and set a patience threshold to stop the process once performance plateaus. This prevents the model from wasting computational resources and protects against the onset of overfitting.
- Class Imbalance Handling: If your dataset contains significantly more "neat" photos than "messy" ones, apply class weights to penalize errors on the minority class more heavily. This ensures the model doesn't become biased toward the more frequent label simply to achieve a high "dummy" accuracy.
- Integrated Gradients for Interpretability: Beyond basic heatmaps, use integrated gradients to attribute the model's prediction to specific input features. This provides a mathematically grounded way to explain why a specific pile of laundry was flagged as "messy" to the end user.
- Transfer Learning Fine-Tuning: Start with a pre-trained architecture like MobileNetV2 and initially freeze the base layers to retain general visual features. Once the top layers are stable, unfreeze the deeper layers and retrain with a very low learning rate to specialize the model for your specific environment.
- Standardized Evaluation Metrics: Supplement your accuracy scores with Precision and Recall to understand the true cost of a false alarm. In a smart-home context, high Precision ensures the user isn't notified about a mess that doesn't exist, while high Recall ensures no actual mess is missed.
- Data Pipeline Optimization: Use the tf.data API to prefetch and batch your data, ensuring the GPU never sits idle while waiting for the CPU to load images. This "pipelining" approach can drastically reduce training time, especially when working with high-resolution datasets.
- Model Versioning and Lineage: Maintain a strict log of hyperparameters and dataset versions for every training run you execute. This allows you to roll back to a previous "best" version if a new architectural change unexpectedly degrades performance in the field.
- Post-Training Pruning: After quantization, consider pruning the model to remove redundant neural connections that contribute little to the final prediction. Hence, this further optimizes the model for edge devices by reducing the number of parameters the processor needs to calculate.

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

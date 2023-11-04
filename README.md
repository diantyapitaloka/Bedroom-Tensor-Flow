## Tensor Flow

Of course, machine learning always needs data. In the initial stage we need to understand the dataset we have first. Some things you need to know are the format of the data, number of samples, and how many labels. Apart from that, we also need to ensure whether the dataset is continuous data (regression problem) or discrete data (classification problem).

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

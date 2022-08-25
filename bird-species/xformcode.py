import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from parallels_plugin import parallels_core
import os


test_df = parallels_core.list("/Users/jitendra/Kaggle/data/Bird-Species/bird_species/test/", "data_input")

model_df = parallels_core.list('/Users/jitendra/Kaggle/data/Bird-Species/bird_species/model/', "model_input")

print(test_df.to_string())
print(test_df['FileName'].to_list()[0])

model_file = parallels_core.get_local_paths(model_df)[0]
print(model_file)
model = keras.models.load_model(model_file)

data_files = parallels_core.get_local_paths(test_df)
data_folder = os.path.dirname(os.path.dirname(data_files[0]))
print(data_folder)
print(data_files)


ds = tf.keras.utils.image_dataset_from_directory(
    data_folder,
    labels = [75,75,75,75,75,139,139,139,139,139,292,292,292,292,292,299,299,299,299,299,399,399,399,399,399],
    shuffle = False,
    batch_size=2,
    image_size=(112,112))

print("DEBUG get dataset iterator")
ds_numpy_iter = tfds.as_numpy(ds) 

print("DEBUG Now iterate over dataset and perform predictions")
correct_labels = 0
num_predictions = 0
batch_num=0
for i in ds_numpy_iter:
    if batch_num % 5 == 0:
        print('Predicting batch {}'.format(batch_num))
    batch_num += 1
    img, label = i
    predictions = model.predict(img)
    predicted_classes = np.argmax(predictions, axis=1)
    correct_labels += np.sum(label == predicted_classes)
    num_predictions += predicted_classes.shape[0]
    
accuracy = float(correct_labels)/num_predictions

print("Number of Predictions = {},\nCorrect Predictions = {},\nAccuracy = {}".format(num_predictions, correct_labels, accuracy))

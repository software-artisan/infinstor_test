import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from concurrent_plugin import concurrent_core
import tempfile
import os

#Preprocess images
def preprocess_image(img_path):
    image = tf.io.read_file(img_path)
    image1 = tf.io.decode_jpeg(image, channels=3)
    image2 = tf.image.resize(image1, [112, 112])
    return image2

def sharpen_image(image):
    image_sharp = tfa.image.sharpness(image, 10.0)
    return image_sharp

print('Environment## 1' , os.environ)
#Make Image Dataset
test_df = concurrent_core.list("/Users/jitendra/Kaggle/data/Bird-Species/bird_species/test/", "data_input")

print('Environment## 2' , os.environ)
data_files = concurrent_core.get_local_paths(test_df)

print('Environment## 3' , os.environ)

print(data_files[0:10])


path_ds = tf.data.Dataset.from_tensor_slices(data_files)

##Preprocessing dataset
image_ds = path_ds.map(preprocess_image)
image_ds = image_ds.map(sharpen_image)

tmpdir = tempfile.mkdtemp()
print(tmpdir)

image_path_ds = tf.data.Dataset.zip((image_ds, path_ds))
image_path_ds = image_path_ds.batch(1)
image_path_iter = tfds.as_numpy(image_path_ds)


for image, path in image_path_iter:
    path = path[0].decode('utf-8')
    img_file_name = os.path.basename(path)
    species_name = os.path.basename(os.path.dirname(path))
    species_name = species_name.replace(' ', '_')
    image = image.reshape((112, 112, 3))
    dst_path = os.path.join(tmpdir, species_name, img_file_name)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    writer = tf.keras.utils.save_img(os.path.join(tmpdir, species_name, img_file_name), image)
    concurrent_core.concurrent_log_artifact(dst_path, os.path.join("output", species_name))

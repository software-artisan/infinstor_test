import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from parallels_plugin import parallels_core
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


#Make Image Dataset
test_df = parallels_core.list("/Users/jitendra/Kaggle/data/Bird-Species/bird_species/test/", "data_input")

data_files = parallels_core.get_local_paths(test_df)

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
    image = image.reshape((112, 112, 3))
    dst_path = os.path.join(tmpdir, species_name, img_file_name)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    writer = tf.keras.utils.save_img(os.path.join(tmpdir, species_name, img_file_name), image)
    parallels_core.parallels_log_artifact(dst_path, os.path.join("output", species_name))
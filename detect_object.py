import tensorflow as tf
from tensorflow.keras import Model
import glob
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model.h5')

# helper function to load data
def read_data(file_path):
  img = tf.io.read_file(file_path)
  img = tf.image.decode_jpeg(img)
  img = tf.image.resize(img, [160, 160], method=tf.image.ResizeMethod.BILINEAR)
  img = tf.cast(img, tf.float32)
  img = (img / 127.5) - 1.0

  return img

# Classes as defined in training
CLASS_NAMES = ['lamp', 'shaver', 'bottle', 'toothbrush']

# load image
file_paths = glob.glob('validation-images/*jpg')

if len(file_paths) == 0:
  print('\n=========================================================================================')
  print('"validation-images" folder is empty, Please put images in this folder to get predictions')
  print('=========================================================================================\n')

for idx, image_path in enumerate(file_paths):
  image = read_data(image_path)
  # print(np.shape(image))
  # plt.figure()
  # plt.imshow(image)
  prediction = model(image[np.newaxis,:])
  wanted_index = np.argmax(prediction.numpy())
  print(prediction)
  print(str(file_paths[idx]) + ' is - ' + str(CLASS_NAMES[wanted_index]))

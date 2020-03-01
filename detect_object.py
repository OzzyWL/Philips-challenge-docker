import tensorflow as tf
from tensorflow.keras import Model
import glob
import numpy as np

IMG_SHAPE = (160,160, 3)

# re-create model
# ==================================================================
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

dummy_mat = np.zeros((1, 160, 160, 3), dtype=float64)
print(np.shape(dummy_mat))

feature_batch = base_model(dummy_mat)
print(feature_batch.shape)
base_model.trainable = True 

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
# I want to predict for 4 classes
prediction_layer = tf.keras.layers.Dense(4)
prediction_batch = prediction_layer(feature_batch_average)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
model.load_weights('weights')
# ==================================================================

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

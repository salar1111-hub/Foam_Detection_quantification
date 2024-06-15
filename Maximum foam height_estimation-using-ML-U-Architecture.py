import tensorflow as tf
from tensorflow.keras import models, layers, backend as K
import os
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from focal_loss import BinaryFocalLoss
from matplotlib import pyplot as plt
def dice_coefficient(y_true, y_pred):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersect = K.sum(y_true_flat * y_pred_flat)
    return (2.0 * intersect + 1.0) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + 1.0)
def jaccard_coefficient(y_true, y_pred):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersect = K.sum(y_true_flat * y_pred_flat)
    return (intersect + 1.0) / (K.sum(y_true_flat) + K.sum(y_pred_flat) - intersect + 1.0)
def jaccard_loss(y_true, y_pred):
    return -jaccard_coefficient(y_true, y_pred)
def dice_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

def convolution_block(input_tensor, filter_size, num_filters, dropout_rate, use_batch_norm=False):
    x = layers.Conv2D(num_filters, (filter_size, filter_size), padding="same")(input_tensor)
    if use_batch_norm:
        x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2D(num_filters, (filter_size, filter_size), padding="same")(x)
    if use_batch_norm:
        x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation("relu")(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)   
    return x
def build_unet_model(input_shape, num_classes=1, dropout_rate=0.0, use_batch_norm=True):
    base_filters = 64
    filter_size = 3
    upsampling_size = 2
    inputs = layers.Input(input_shape, dtype=tf.float32)

    conv1 = convolution_block(inputs, filter_size, base_filters, dropout_rate, use_batch_norm)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = convolution_block(pool1, filter_size, base_filters * 2, dropout_rate, use_batch_norm)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = convolution_block(pool2, filter_size, base_filters * 4, dropout_rate, use_batch_norm)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = convolution_block(pool3, filter_size, base_filters * 8, dropout_rate, use_batch_norm)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = convolution_block(pool4, filter_size, base_filters * 16, dropout_rate, use_batch_norm)

    up6 = layers.UpSampling2D(size=(upsampling_size, upsampling_size))(conv5)
    up6 = layers.concatenate([up6, conv4], axis=3)
    conv6 = convolution_block(up6, filter_size, base_filters * 8, dropout_rate, use_batch_norm)

    up7 = layers.UpSampling2D(size=(upsampling_size, upsampling_size))(conv6)
    up7 = layers.concatenate([up7, conv3], axis=3)
    conv7 = convolution_block(up7, filter_size, base_filters * 4, dropout_rate, use_batch_norm)

    up8 = layers.UpSampling2D(size=(upsampling_size, upsampling_size))(conv7)
    up8 = layers.concatenate([up8, conv2], axis=3)
    conv8 = convolution_block(up8, filter_size, base_filters * 2, dropout_rate, use_batch_norm)

    up9 = layers.UpSampling2D(size=(upsampling_size, upsampling_size))(conv8)
    up9 = layers.concatenate([up9, conv1], axis=3)
    conv9 = convolution_block(up9, filter_size, base_filters, dropout_rate, use_batch_norm)

    outputs = layers.Conv2D(num_classes, kernel_size=(1, 1))(conv9)
    outputs = layers.BatchNormalization(axis=3)(outputs)
    outputs = layers.Activation('sigmoid')(outputs)  # Use 'softmax' for multiple classes

    model = models.Model(inputs, outputs, name="UNet")
    print(model.summary())
    return model
'''   
The main idea for above parts is from the links below but completely modified and coded in a different way
https://github.com/MoleImg/Attention_UNet/blob/master/AttResUNet.py
https://github.com/bnsreenu/python_for_microscopists
'''
image_directory = './Images-F/'
mask_directory = './Masks-F/'
SIZE = 256

image_dataset = []
images = os.listdir(image_directory)
for image_name in images:
    if image_name.split('.')[1] == 'tif':
        image = cv2.imread(os.path.join(image_directory, image_name), 1)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

mask_dataset = []
masks = os.listdir(mask_directory)
for image_name in masks:
    if image_name.split('.')[1] == 'tif':
        mask = cv2.imread(os.path.join(mask_directory, image_name), 0)
        mask = Image.fromarray(mask)
        mask = mask.resize((SIZE, SIZE))
        mask_dataset.append(np.array(mask))

image_dataset = np.array(image_dataset) / 255.0
mask_dataset = np.expand_dims(np.array(mask_dataset), 3) / 255.0

X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.10, random_state=0)
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
batch_size = 8
unet_model = build_unet_model(input_shape)
unet_model.compile(optimizer=Adam(lr=1e-3), loss=BinaryFocalLoss(gamma=2), metrics=['accuracy', jaccard_coefficient])
start_time = datetime.now()
unet_history = unet_model.fit(X_train, y_train,
                              batch_size=batch_size,
                              epochs=100,
                              verbose=1,
                              validation_data=(X_test, y_test),
                              shuffle=False)
end_time = datetime.now()
execution_time = end_time - start_time
unet_model.save('./FModel.hdf5')

model_path = "./Model.hdf5"
model = tf.keras.models.load_model(model_path, compile=False)

test_img_other = cv2.imread('./Stack0022.tif')
test_img_other = cv2.resize(test_img_other, (256, 256)) / 255
test_img_other_input = np.expand_dims(test_img_other, 0)
prediction_other = (model.predict(test_img_other_input)[0, :, :, 0] > 0.5).astype(np.uint8)
plt.imshow(prediction_other)
plt.show()
y = 260
x = 0
h = 1000
w = 1024
PPmm = 31 / 0.01
img = cv2.resize(prediction_other, (1280, 1024))[x:w, y:h]
plt.imshow(img, 'gray')
height, width = img.shape
width_cutoff = width // 12
height_sum = 0
for i in range(12):
    s = img[:, i * width_cutoff:(i + 1) * width_cutoff]
    cnts_s = cv2.findContours(s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    cnt_s = max(cnts_s, key=cv2.contourArea)
    _, _, _, h_si = cv2.boundingRect(cnt_s)
    height_sum += h_si

average_height = height_sum * 1 / (PPmm * 12)
print('Height_Average:', average_height)

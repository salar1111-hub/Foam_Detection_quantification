import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from datetime import datetime
from focal_loss import BinaryFocalLoss

def dice_coefficient(y_true, y_pred):
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_pred)
    intersect = tf.keras.backend.sum(y_true_flat * y_pred_flat)
    return (2.0 * intersect + 1.0) / (tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat) + 1.0)
def jaccard_coefficient(y_true, y_pred):
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_pred)
    intersect = tf.keras.backend.sum(y_true_flat * y_pred_flat)
    return (intersect + 1.0) / (tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat) - intersect + 1.0)
def jaccard_loss(y_true, y_pred):
    return -jaccard_coefficient(y_true, y_pred)
def dice_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)
def repeat_elements(tensor, rep):
    return layers.Lambda(lambda x, repnum: tf.keras.backend.repeat_elements(x, repnum, axis=3),
                         arguments={'repnum': rep})(tensor)
def gating_signal(input, out_size, batch_norm=False):
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x
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
def attention_block(x, gating, inter_shape):
    shape_x = tf.keras.backend.int_shape(x)
    shape_g = tf.keras.backend.int_shape(gating)

    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
    shape_theta_x = tf.keras.backend.int_shape(theta_x)
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                        strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                        padding='same')(phi_g)
    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = tf.keras.backend.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)
    upsample_psi = repeat_elements(upsample_psi, shape_x[3])
    y = layers.multiply([upsample_psi, x])
    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn
def Attention_UNet(input_shape, num_classes=1, dropout_rate=0.0, batch_norm=True):
    FILTER_NUM = 64
    FILTER_SIZE = 3 
    UP_SAMP_SIZE = 2   
    inputs = layers.Input(input_shape, dtype=tf.float32)
    conv_128 = convolution_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    conv_64 = convolution_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    conv_32 = convolution_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    conv_16 = convolution_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    conv_8 = convolution_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = convolution_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = convolution_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=3)
    up_conv_64 = convolution_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = convolution_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    conv_final = layers.Conv2D(num_classes, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)
    model = models.Model(inputs, conv_final, name="Attention_UNet")
    return model
'''   
The main idea for above parts is from the links below but completely modified and coded in a different way
https://github.com/MoleImg/Attention_UNet/blob/master/AttResUNet.py
https://github.com/bnsreenu/python_for_microscopists
'''
image_directory = './Images/'
mask_directory = './Masks/'
SIZE = 256
image_dataset = []
mask_dataset = []
images = [img for img in os.listdir(image_directory) if img.endswith('.tif')]
masks = [mask for mask in os.listdir(mask_directory) if mask.endswith('.tif')]

for image_name in images:
    image = cv2.imread(os.path.join(image_directory, image_name), 1)
    image = Image.fromarray(image)
    image = image.resize((SIZE, SIZE))
    image_dataset.append(np.array(image))
for mask_name in masks:
    mask = cv2.imread(os.path.join(mask_directory, mask_name), 0)
    mask = Image.fromarray(mask)
    mask = mask.resize((SIZE, SIZE))
    mask_dataset.append(np.expand_dims(np.array(mask), 3))
image_dataset = np.array(image_dataset) / 255.0
mask_dataset = np.array(mask_dataset) / 255.0

X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.10, random_state=0)

input_shape = X_train.shape[1:]

batch_size = 8
att_unet_model = Attention_UNet(input_shape)
att_unet_model.compile(optimizer=Adam(lr=1e-3), loss=BinaryFocalLoss(gamma=2), metrics=['accuracy', jaccard_coefficient])
start_time = datetime.now()

att_unet_model = Attention_UNet(input_shape)
att_unet_model.compile(optimizer=Adam(lr=1e-3), loss=BinaryFocalLoss(gamma=2), metrics=['accuracy', jaccard_coefficient])
start2 = datetime.now() 
att_unet_history = att_unet_model.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=100)
stop2 = datetime.now()
execution_time_Att_Unet = stop2-start2
att_unet_model.save('./Model.hdf5')

from matplotlib import pyplot as plt
model_path = "./Model.hdf5"
model = tf.keras.models.load_model(model_path, compile=False)
test_img_other = cv2.imread('./Stack0022.tif')
test_img_other = cv2.resize(test_img_other, (256, 256)) / 255
test_img_other_input = np.expand_dims(test_img_other, 0)
prediction_other = (model.predict(test_img_other_input)[0, :, :, 0] > 0.5).astype(np.uint8)
plt.imshow(prediction_other)
plt.show()
y=260
x=0
h=1000
w=1024
PPmm = 31 / 0.01
img = cv2.resize(prediction_other, (1280, 1024))[x:w, y:h]
plt.imshow(img,'gray')
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
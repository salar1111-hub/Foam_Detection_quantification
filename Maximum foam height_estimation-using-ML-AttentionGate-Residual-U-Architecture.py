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
def dice_coefficient(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    return (2.0 * intersection + 1.0) / (K.sum(y_true) + K.sum(y_pred) + 1.0)
def jaccard_coefficient(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    return (intersection + 1.0) / (K.sum(y_true) + K.sum(y_pred) - intersection + 1.0)
def jaccard_loss(y_true, y_pred):
    return -jaccard_coefficient(y_true, y_pred)
def dice_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)
# Utility function for repeating elements
def repeat_elements(tensor, rep):
    return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                         arguments={'repnum': rep})(tensor)
def attention_block(x, gating, inter_shape):
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                        strides=(2, 2), padding='same')(phi_g)
    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    upsample_psi = layers.UpSampling2D(size=(2, 2))(sigmoid_xg)
    upsample_psi = repeat_elements(upsample_psi, K.int_shape(x)[3])
    y = layers.multiply([upsample_psi, x])
    result = layers.Conv2D(K.int_shape(x)[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn
def res_conv_block(x, filter_size, size, dropout, batch_norm=False):
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv) 
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)
    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)
    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation('relu')(res_path)
    return res_path
def gating_signal(input, out_size, batch_norm=False):
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x
def Attention_ResUNet(input_shape, num_classes=1, dropout_rate=0.0, batch_norm=True):
    filter_num = 64
    filter_size = 3
    up_samp_size = 2
    inputs = layers.Input(input_shape, dtype=tf.float32)
    axis = 3 
    conv_128 = res_conv_block(inputs, filter_size, filter_num, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)
    conv_64 = res_conv_block(pool_64, filter_size, 2 * filter_num, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)
    conv_32 = res_conv_block(pool_32, filter_size, 4 * filter_num, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)
    conv_16 = res_conv_block(pool_16, filter_size, 8 * filter_num, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2, 2))(conv_16)
    conv_8 = res_conv_block(pool_8, filter_size, 16 * filter_num, dropout_rate, batch_norm)
    
    gating_16 = gating_signal(conv_8, 8 * filter_num, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8 * filter_num)
    up_16 = layers.UpSampling2D(size=(up_samp_size, up_samp_size))(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=axis)
    up_conv_16 = res_conv_block(up_16, filter_size, 8 * filter_num, dropout_rate, batch_norm)
    
    gating_32 = gating_signal(up_conv_16, 4 * filter_num, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4 * filter_num)
    up_32 = layers.UpSampling2D(size=(up_samp_size, up_samp_size))(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=axis)
    up_conv_32 = res_conv_block(up_32, filter_size, 4 * filter_num, dropout_rate, batch_norm)
    
    gating_64 = gating_signal(up_conv_32, 2 * filter_num, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2 * filter_num)
    up_64 = layers.UpSampling2D(size=(up_samp_size, up_samp_size))(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=axis)
    up_conv_64 = res_conv_block(up_64, filter_size, 2 * filter_num, dropout_rate, batch_norm)
    
    gating_128 = gating_signal(up_conv_64, filter_num, batch_norm)
    att_128 = attention_block(conv_128, gating_128, filter_num)
    up_128 = layers.UpSampling2D(size=(up_samp_size, up_samp_size))(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=axis)
    up_conv_128 = res_conv_block(up_128, filter_size, filter_num, dropout_rate, batch_norm)
    
    conv_final = layers.Conv2D(num_classes, kernel_size=(1, 1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=axis)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)
    
    model = models.Model(inputs, conv_final, name="AttentionResUNet")
    return model
'''   
The main idea for above parts is from the links below but completely modified and coded in a different way
https://github.com/MoleImg/Attention_UNet/blob/master/AttResUNet.py
https://github.com/bnsreenu/python_for_microscopists
'''
def load_data(image_directory, mask_directory, size):
    image_dataset = []
    mask_dataset = []
    images = [img for img in os.listdir(image_directory) if img.endswith('.tif')]
    masks = [mask for mask in os.listdir(mask_directory) if mask.endswith('.tif')]  
    for image_name in images:
        image = cv2.imread(os.path.join(image_directory, image_name), 1)
        image = Image.fromarray(image)
        image = image.resize((size, size))
        image_dataset.append(np.array(image))
    for mask_name in masks:
        mask = cv2.imread(os.path.join(mask_directory, mask_name), 0)
        mask = Image.fromarray(mask)
        mask = mask.resize((size, size))
        mask_dataset.append(np.array(mask))   
    return np.array(image_dataset) / 255.0, np.expand_dims(np.array(mask_dataset), 3) / 255.0
def main():
    image_directory = './Images/'
    mask_directory = './Masks/'
    SIZE = 256
    X, y = load_data(image_directory, mask_directory, SIZE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
    input_shape = X_train.shape[1:]
    batch_size = 8
    att_res_unet_model = Attention_ResUNet(input_shape)
    att_res_unet_model.compile(optimizer=Adam(lr=1e-3), loss=BinaryFocalLoss(gamma=2),
                               metrics=['accuracy', dice_coefficient])
    start_time = datetime.now()
    att_res_unet_history = att_res_unet_model.fit(X_train, y_train,
                                                  verbose=1,
                                                  batch_size=batch_size,
                                                  validation_data=(X_test, y_test),
                                                  shuffle=False,
                                                  epochs=100)
    end_time = datetime.now()
    execution_time = end_time - start_time

    att_res_unet_model.save('./Model.hdf5')

if __name__ == "__main__":
    main
    

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


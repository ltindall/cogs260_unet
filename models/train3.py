from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, Dropout, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, BatchNormalization, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 128
img_cols = 128

orig_img_rows = 420
orig_img_cols = 580


smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    #conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1, training=False)
    #conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    #conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1, training=False)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    #conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2, training=False)
    #conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    #conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2, training=False)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    #conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3, training=False)
    #conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    #conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3, training=False)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    #conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4, training=False)
    #conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    #conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4, training=False)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    #conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5, training=False)
    #conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    #conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5, training=False)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    #conv6 = BatchNormalization(axis=3)(conv6)
    #conv6 = Activation('relu')(conv6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    #conv6 = BatchNormalization(axis=3)(conv6)
    #conv6 = Activation('relu')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    #conv7 = BatchNormalization(axis=3)(conv7)
    #conv7 = Activation('relu')(conv7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    #conv7 = BatchNormalization(axis=3)(conv7)
    #conv7 = Activation('relu')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    #conv8 = BatchNormalization(axis=3)(conv8)
    #conv8 = Activation('relu')(conv8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    #conv8 = BatchNormalization(axis=3)(conv8)
    #conv8 = Activation('relu')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    #conv9 = BatchNormalization(axis=3)(conv9)
    #conv9 = Activation('relu')(conv9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    #conv9 = BatchNormalization(axis=3)(conv9)
    #conv9 = Activation('relu')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)
        #imgs_p[i] = imgs[i]

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def postprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        img = (imgs[i] > 0.5).astype('uint8')
        imgs_p[i] = resize(img, (orig_img_rows, orig_img_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=16, nb_epoch=40, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    imgs_mask_test = preprocess(imgs_mask_test)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

if __name__ == '__main__':
    train_and_predict()

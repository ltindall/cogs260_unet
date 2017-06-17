from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 96
img_cols = 96

orig_img_rows = 420
orig_img_cols = 580


smooth = 1.

np.random.seed(7)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = ELU()(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(conv1)
    conv1 = ELU()(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #pool1 = Dropout(0.2)(pool1)

    conv2 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = ELU()(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(conv2)
    conv2 = ELU()(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #pool2 = Dropout(0.2)(pool2)

    conv3 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = ELU()(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(conv3)
    conv3 = ELU()(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #pool3 = Dropout(0.2)(pool3)

    conv4 = Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = ELU()(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same')(conv4)
    conv4 = ELU()(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #pool4 = Dropout(0.2)(pool4)

    conv5 = Conv2D(1024, (3, 3), kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = ELU()(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, (3, 3), kernel_initializer='he_normal', padding='same')(conv5)
    conv5 = ELU()(conv5)
    conv5 = BatchNormalization()(conv5)
    #conv5 = Dropout(0.2)(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), padding='same')(up6)
    conv6 = ELU()(conv6)
    #conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, (3, 3), padding='same')(conv6)
    conv6 = ELU()(conv6)
    #conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.2)(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), padding='same')(up7)
    conv7 = ELU()(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, (3, 3), padding='same')(conv7)
    conv7 = ELU()(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(0.2)(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), padding='same')(up8)
    conv8 = ELU()(conv8)
    #conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, (3, 3), padding='same')(conv8)
    conv8 = ELU()(conv8)
    #conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(0.2)(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), padding='same')(up9)
    conv9 = ELU()(conv9)
    #conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, (3, 3), padding='same')(conv9)
    conv9 = ELU()(conv9)
    #conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(0.2)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)
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

    print('imgs_train = ',len(imgs_train))
    print('imgs_mask_train = ',len(imgs_mask_train))

    shuffle = np.random.permutation(len(imgs_train))
    imgs_train = imgs_train[shuffle]
    imgs_mask_train = imgs_mask_train[shuffle]

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
    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=40, verbose=1, shuffle=True,
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

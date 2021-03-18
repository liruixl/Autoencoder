
import os
import keras
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import  Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils

from PIL import Image

# from keras.utils import plot_model

from keras.callbacks import LearningRateScheduler
import keras.backend as K


def load_banklogo_data():
    img_list = []
    anoimg_list = []
    bank_dir = r'E:\CheckDS\bank_logo0927'
    anomaly_dir = r'E:\CheckDS\bank_logo_anomaly'
    anomaly_dir = r'E:\CheckDS\bank_hualing_anomaly'

    frames = sorted(os.listdir(bank_dir))
    for f in frames:
        imgpath = os.path.join(bank_dir, f)
        img = np.array(Image.open(imgpath))
        img_list.append(img)

    frames = sorted(os.listdir(anomaly_dir))
    for f in frames:
        imgpath = os.path.join(anomaly_dir, f)
        img = np.array(Image.open(imgpath))
        anoimg_list.append(img)

    # anoimg_list = anoimg_list[:10]
    return np.array(img_list), np.array(anoimg_list)



(X_train, X_test) = load_banklogo_data()

X_train = X_train.reshape(X_train.shape[0], 96, 96, 3)
X_test = X_test.reshape(X_test.shape[0], 96, 96, 3)

X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


x = Input(shape=(96, 96, 3))

# Encoder
conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
conv1_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D((2, 2), padding='same')(conv1_2)
conv1_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)
h = MaxPooling2D((2, 2), padding='same')(conv1_3)


# Decoder
conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
up1 = UpSampling2D((2, 2))(conv2_1)
conv2_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D((2, 2))(conv2_2)
conv2_3 = Conv2D(16, (3, 3), activation='relu', padding='same')(up2)
up3 = UpSampling2D((2, 2))(conv2_3)
r = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up3)

autoencoder = Model(inputs=x, outputs=r)
# autoencoder.load_weights("../model/banklogo_1000_0.51.h5")  # model/banklogo_1000_0.51.h5   郭男bad模型
autoencoder.load_weights("../model/banklogo_hualing_bad.h5")  # 华菱 bad模型

# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.summary()



decoded_imgs = autoencoder.predict(X_test)


save_dir = r'E:\CheckDS\bank_huanling_decoder_bad'
save_en = '{}_en.jpg'
save_de = '{}_de.jpg'

for i in range(len(decoded_imgs)):
    X_test[i] = X_test[i] * 255
    decoded_imgs[i] = decoded_imgs[i] * 255

    im1 = Image.fromarray(X_test[i].astype(np.uint8))
    im2 = Image.fromarray(decoded_imgs[i].astype(np.uint8))

    im1.save(os.path.join(save_dir, save_en.format(i)))
    im2.save(os.path.join(save_dir, save_de.format(i)))
    print('保存第{}张', i)

n = 10
plt.figure(figsize=(20, 6))

for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

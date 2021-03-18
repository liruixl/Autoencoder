
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


# 1 修改数据加载
# 2 修改是否训练
# 3 修改模型路径 save load

def load_banklogo_data():
    img_list = []
    bank_dir = r'E:\CheckDS\bank_hualing'
    frames = sorted(os.listdir(bank_dir))
    for f in frames:
        imgpath = os.path.join(bank_dir, f)
        img = np.array(Image.open(imgpath))
        img_list.append(img)

    test_img_list = img_list[20:30]
    return np.array(img_list), np.array(test_img_list)


def load_banklogo_anomaly_data():
    img_list = []
    anoimg_list = []
    bank_dir = r'E:\CheckDS\bank_logo0927'
    anomaly_dir = r'E:\CheckDS\bank_logo_anomaly'

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

    # anoimg_list = anoimg_list[-10:]
    # anoimg_list = anoimg_list[:10]

    return np.array(img_list), np.array(anoimg_list)


(X_train, X_test) = load_banklogo_data()

# (X_train, X_test) = load_banklogo_anomaly_data()


X_train = X_train.reshape(X_train.shape[0], 96, 96, 3)
X_test = X_test.reshape(X_test.shape[0], 96, 96, 3)

X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


x = Input(shape=(96, 96, 3))

# Encoder
conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D((2, 2), padding='same')(conv1_2)
conv1_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
pool3 = MaxPooling2D((2, 2), padding='same')(conv1_3)
conv1_4 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool3)

h = MaxPooling2D((2, 2), padding='same')(conv1_4)


# Decoder
conv2_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(h)
up1 = UpSampling2D((2, 2))(conv2_1)
conv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D((2, 2))(conv2_2)
conv2_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
up3 = UpSampling2D((2, 2))(conv2_3)
conv2_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up3)
up4 = UpSampling2D((2, 2))(conv2_4)

r = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up4)

autoencoder = Model(inputs=x, outputs=r)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

# plot_model(autoencoder, to_file='model.png')



epochs = 100
batch_size = 8

checkpoint_path = "../model/banklogo_huangling_1.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个保存模型权重的回调
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,save_best_only=True)

def scheduler(epoch):
    # 每隔300个epoch，学习率减小为原来的1/10
    if epoch % 300 == 0 and epoch != 0:
        lr = K.get_value(autoencoder.optimizer.lr)
        K.set_value(autoencoder.optimizer.lr, lr * 0.1)
        print("=========lr*0.1==========lr changed to {}".format(lr * 0.1))
    return K.get_value(autoencoder.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)



history = autoencoder.fit(X_train, X_train, batch_size=batch_size, epochs=epochs,
                          verbose=1, validation_data=(X_test, X_test),
                          callbacks=[cp_callback, reduce_lr], shuffle=True)


decoded_imgs = autoencoder.predict(X_test)

# save_dir = r'E:\CheckDS\bank_logo_decoder'
# save_en = '{}_en.jpg'
# save_de = '{}_de.jpg'
#
# for i in range(len(decoded_imgs)):
#     X_test[i] = X_test[i] * 255
#     decoded_imgs[i] = decoded_imgs[i] * 255
#
#     im1 = Image.fromarray(X_test[i].astype(np.uint8))
#     im2 = Image.fromarray(decoded_imgs[i].astype(np.uint8))
#
#     im1.save(os.path.join(save_dir, save_en.format(i)))
#     im2.save(os.path.join(save_dir, save_de.format(i)))
#     print('保存第{}张', i)


n = 10
plt.figure(figsize=(20, 6))  # 2000*600

for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(decoded_imgs[i])  # reshape(28, 28)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

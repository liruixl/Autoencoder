

import cv2
import os
import numpy as np
from skimage import io, transform,color
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from skimage import filters

def rotate(image, angle, center=None, scale=1.0):
    # Grab the dimensions of the image
    (h, w) = image.shape[:2]

    # If the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Return the rotated image
    return rotated

def dbd(img,k):
    mean = np.mean(img)

    img = mean + (img-mean)*k

    img[img > 255] = 255
    img[img < 0] = 0

    return img.astype(np.uint8)


def color2color(img, hk, sk):
    hsv = color.rgb2hsv(img)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    h = h*hk
    s = s*sk
    h[h>1] = 1.0
    s[s>1] = 1.0
    hsv[:,:,0] = h
    hsv[:,:,1] = s

    return color.hsv2rgb(hsv)


image = io.imread(r'E:\CheckDS\data_zg\15-41-51.jpg', as_gray=False)

print(image.dtype)

# im1 = exposure.adjust_gamma(image, 2)   #调暗
# im2 = exposure.adjust_gamma(image, 0.5)  #调亮

# im1 = dbd(image,0.8)
# im2 = dbd(image,1.2)

# im1 = transform.rotate(image, 2)
# im2 = transform.rotate(image, -2)

# im1 = rotate(image,2)
# im2 = rotate(image,-2)

im1 = color2color(image,1.05,1.2)
im2 = color2color(image,0.8,1.2)

save_dir = r'E:\CheckDS\data_zg'
save_en = 'hsv_{}.jpg'
save_de = 'hsv_{}.jpg'

# im1 = Image.fromarray(X_test[i].astype(np.uint8))
# im2 = Image.fromarray(decoded_imgs[i].astype(np.uint8))
# im1.save(os.path.join(save_dir,save_en.format(120)))
# im2.save(os.path.join(save_dir,save_de.format(80)))


io.imsave(os.path.join(save_dir,save_en.format(20)), im1)
io.imsave(os.path.join(save_dir,save_en.format(10)), im2)


plt.figure('adjust_gamma',figsize=(8,8))

plt.subplot(131)
plt.title('origin image')
plt.imshow(image,plt.cm.gray)
plt.axis('off')

plt.subplot(132)
plt.title('gamma=2')
plt.imshow(im1,plt.cm.gray)
plt.axis('off')

plt.subplot(133)
plt.title('gamma=0.5')
plt.imshow(im2,plt.cm.gray)
plt.axis('off')

plt.show()
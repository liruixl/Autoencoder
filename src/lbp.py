from skimage import data, io, filters


from skimage import io
from skimage import filters

import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy

import matplotlib.pyplot as plt

from skimage.segmentation import felzenszwalb
from skimage.data import coffee


from skimage.measure import compare_ssim

img_en = io.imread(r'E:\CheckDS\bank_logo_decoder\5_en.jpg', as_gray=True)
img_en_filter = filters.gaussian(img_en,sigma=0.5)   #sigma=5


img_de = io.imread(r'E:\CheckDS\bank_logo_decoder\5_de.jpg', as_gray=True)



# for colour_channel in (0, 1, 2):
#     img[:, :, colour_channel] = skimage.feature.local_binary_pattern(
#         img[:, :, colour_channel], 8,1.0,method='var')

img_en_lbp = skimage.feature.local_binary_pattern(img_en, 8, 1.0, method='var')
img_de_lbp = skimage.feature.local_binary_pattern(img_de, 8, 1.0, method='var')

gray_diff = abs(img_en_filter - img_de)
lbp_diff = abs(img_en_lbp - img_de_lbp)

(score,diff)=compare_ssim(img_en_filter,img_de,full=True)
diff = (diff*255).astype("uint8")
diff = - diff + 255

plt.figure(figsize=(10, 10))  #

ax1 = plt.subplot(2,2,1)  # 一行两列
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
plt.imshow(img_en, cmap='gray')

ax2 = plt.subplot(2,2,2)
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
plt.imshow(img_en_lbp, cmap='gray')

ax3 = plt.subplot(2,2,3)
plt.axis('off')  #去掉坐标轴
plt.imshow(img_de, cmap='gray')


ax4 = plt.subplot(2,2,4)
plt.axis('off')  #去掉坐标轴
plt.imshow(img_de_lbp, cmap='gray')

plt.show()

plt.figure()
plt.subplot(2,2,1)
plt.imshow(gray_diff)
plt.subplot(2,2,2)  # 一行两列
plt.imshow(lbp_diff)
plt.subplot(2,2,3)  # 一行两列
plt.imshow(diff)
plt.show()
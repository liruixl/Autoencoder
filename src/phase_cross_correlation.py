import numpy as np
import matplotlib.pyplot as plt

from skimage import data, draw
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi

image = data.camera()
shift = (-22, 13)

rr1, cc1 = draw.ellipse(259, 248, r_radius=125, c_radius=100,
                        shape=image.shape)

rr2, cc2 = draw.ellipse(300, 200, r_radius=110, c_radius=180,
                        shape=image.shape)

mask1 = np.zeros_like(image, dtype=bool)
mask2 = np.zeros_like(image, dtype=bool)
mask1[rr1, cc1] = True
mask2[rr2, cc2] = True

offset_image = ndi.shift(image, shift)
image *= mask1
offset_image *= mask2

print(f"Known offset (row, col): {shift}")

detected_shift = phase_cross_correlation(image, offset_image,
                                         reference_mask=mask1,
                                         moving_mask=mask2)

print(f"Detected pixel offset (row, col): {-detected_shift}")

fig = plt.figure(figsize=(9,3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)



ax1.imshow(image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offset_image.real, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Masked, offset image')


ax3.imshow(mask1, cmap='gray')
ax3.set_axis_off()
ax3.set_title('Mask1')

plt.show()
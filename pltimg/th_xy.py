import numpy as np
import matplotlib.pyplot as plt


# x = np.random.rand(N)
# y = np.random.rand(N)

xmin, ymin = 1,1
xmax, ymax = 420, 170  # 400-420 160-170


x1 = np.random.randint(400,410,size=(200,))
y1 = np.random.randint(160,170,size=(200,))

x2 = np.random.randint(400, 420,size=(100,))
y2 = np.random.randint(160,170,size=(100,))




a = np.zeros(shape=(300,), dtype=np.int)
b = np.zeros(shape=(300,), dtype=np.int)

a[:200] = x1
a[200:300] = x2


b[:200] = y1
b[200:300] = y2



colors = np.random.rand(300)
# area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

plt.xlabel('x')
plt.ylabel('y')
plt.scatter([1], [1], s=2, c=1, alpha=1)
plt.scatter(a, b, s=2, c=colors, alpha=1)
plt.show()
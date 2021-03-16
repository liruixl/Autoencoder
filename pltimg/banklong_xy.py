import numpy as np
import matplotlib.pyplot as plt


# x = np.random.rand(N)
# y = np.random.rand(N)

xmin, ymin = 1090, 30
xmax, ymax = 1190, 130


x1 = np.random.randint(xmin - 5,xmin + 5,size=(200,))
y1 = np.random.randint(ymin - 5,ymin + 5,size=(200,))
x2 = np.random.randint(xmin - 10,xmin + 10,size=(100,))
y2 = np.random.randint(ymin - 8,ymin + 8,size=(100,))




xx1 = np.random.randint(xmax - 5,xmax + 5,size=(200,))
yy1 = np.random.randint(ymax -5, ymax + 5,size=(200,))
xx2 = np.random.randint(xmax - 10,xmax + 10,size=(100,))
yy2 = np.random.randint(ymax -8, ymax + 8,size=(100,))

a = np.zeros(shape=(600,), dtype=np.int)
b = np.zeros(shape=(600,), dtype=np.int)

a[:200] = x1
a[200:300] = x2
a[300:500] = xx1
a[500:] = xx2

b[:200] = y1
b[200:300] = y2
b[300:500] = yy1
b[500:] = yy2


colors = np.random.rand(600)
# area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(a, b, s=2, c=colors, alpha=1)
plt.show()
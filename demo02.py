# 这个文件用于测试 plt 的画图的显示问题
import matplotlib.pyplot as plt
import numpy as np

##############################画一个简单的图形#####################################
# 首先通过 np.linspace 方式生成 x，
# 它包含了 50 个元素的数组，这 50 个元素均匀的分布在 [0, 2*pi] 的区间上。然后通过 np.sin(x) 生成 y。

x = np.linspace(0, 2 * np.pi, 50)
y = np.sin(x)
# 有了 x 和 y 数据之后，我们通过 plt.plot(x, y) 来画出图形，并通过 plt.show() 来显示。
plt.plot(x, y)
plt.show()
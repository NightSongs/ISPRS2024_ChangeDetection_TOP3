import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from osgeo import gdal
import glob
from matplotlib import font_manager, rc

# 设置中文字体
font_path = r'C:\Windows\Fonts/simsun.ttc'
prop = font_manager.FontProperties(fname=font_path)
rc('font', family='SimSun')

# 初始化每个类的数目
background = 0
building_num = 0
bare_num = 0
water_num = 0
highway_num = 0
railway_num = 0
park_num = 0
change_num = 0

label_paths = glob.glob(r'D:\competition\ISPRS2024\data\gt/*.tif')

for label_path in label_paths:
    label = gdal.Open(label_path).ReadAsArray()
    background += np.sum(label == 0)
    building_num += np.sum(label == 200)
    bare_num += np.sum(label == 150)
    water_num += np.sum(label == 100)
    highway_num += np.sum(label == 250)
    railway_num += np.sum(label == 220)
    park_num += np.sum(label == 50)
    change_num += np.sum(label > 0)

# classes = ('建筑物', '推堆土', '库塘水面', '公路', '铁路', '公园')
# numbers = [building_num, bare_num, water_num, highway_num, railway_num, park_num]
classes = ('非变化', '变化')
numbers = [background, change_num]

# 使用seaborn绘图
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=classes, y=numbers, palette="Spectral")

# 设置标题和坐标轴标签
plt.title('变化类别样本量可视化', fontproperties=prop, fontsize=16)
plt.xlabel('类别', fontproperties=prop, fontsize=12)
plt.ylabel('样本数量', fontproperties=prop, fontsize=12)

# 添加数值标签
for i, v in enumerate(numbers):
    ax.text(i, v, str(v), color='black', ha='center', va='bottom', fontproperties=prop, fontsize=10)

# 保存图片
if not os.path.exists(r"D:\competition\ISPRS2024\决赛\答辩"):
    os.makedirs(r"D:\competition\ISPRS2024\决赛\答辩")
plt.savefig(r"D:\competition\ISPRS2024\决赛\答辩/变化类别样本量可视化.png", dpi=600, bbox_inches="tight")
plt.show()

import os

from lidar_platform import cc

dir_ = r'C:\DATA\AlexAndreault'

file = os.path.join(dir_, 'Loire_zone18-3_C3_696000_6680000_class.laz')

#%%
res = cc.density(file, 1, 'knn', verbose=True)

import numpy as np

# 读取npy文件
data = np.load('ref_motion.npy', allow_pickle=True)

# 查看内容
print(data)

# 查看数据类型和形状
print(f"数据类型: {data.dtype}")
print(f"数组形状: {data.shape}")
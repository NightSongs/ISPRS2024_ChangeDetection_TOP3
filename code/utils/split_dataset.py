import os
from sklearn.model_selection import KFold

# 设置目录路径和文件名前缀
directory = '/mnt/c/dataset_tmp/dataset/T1'

# 获取目录下的所有图像文件
files = [f.split('.')[0] for f in os.listdir(directory) if f.endswith('.tif') or f.endswith('.png')]

# 使用5-fold交叉验证进行分组
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 生成每组的训练集和验证集，并保存到txt文件中
for fold, (train_index, val_index) in enumerate(kf.split(files)):
    train_files = [files[i] for i in train_index]
    val_files = [files[i] for i in val_index]

    os.makedirs(f'/mnt/c/dataset_tmp/dataset/split/fold{fold+1}')

    # 保存训练集到txt文件
    train_filename = f'/mnt/c/dataset_tmp/dataset/split/fold{fold+1}/train.txt'
    with open(train_filename, 'w') as f:
        f.write('\n'.join(train_files))

    # 保存验证集到txt文件
    val_filename = f'/mnt/c/dataset_tmp/dataset/split/fold{fold+1}/val.txt'
    with open(val_filename, 'w') as f:
        f.write('\n'.join(val_files))
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取处理后的数据集
df = pd.read_csv('../dataset/processed_dataset.csv')

# 将数据集划分为训练集、验证集和测试集
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

train_data, test_data = train_test_split(df, test_size=test_ratio, shuffle=False)
train_data, val_data = train_test_split(train_data, test_size=val_ratio/(train_ratio+val_ratio), shuffle=False)

# 保存划分后的数据集为CSV文件
train_data.to_csv('../dataset/train_dataset.csv', index=False)
val_data.to_csv('../dataset/val_dataset.csv', index=False)
test_data.to_csv('../dataset/test_dataset.csv', index=False)
print('数据划分已完成')
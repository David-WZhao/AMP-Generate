import pandas as pd

# 读取CSV文件
df = pd.read_csv('pre_train_clean.csv')

# 确保 'sequence' 列是字符串类型
df['Sequence'] = df['Sequence'].astype(str)

# 计算序列长度
df['length'] = df['Sequence'].apply(len)

# 统计每个长度的序列数量
length_counts = df['length'].value_counts().sort_index()

# 输出结果
print("每个长度的序列数量：")
print(length_counts)
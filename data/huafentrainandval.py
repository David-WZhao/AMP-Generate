import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
df = pd.read_csv('pre_train_clean.csv')

# 确保 'Sequence' 列是字符串类型
df['Sequence'] = df['Sequence'].astype(str)

# 计算序列长度
df['length'] = df['Sequence'].apply(len)

# 筛选长度在 5 到 50 之间的序列
df_filtered = df[(df['length'] >= 5) & (df['length'] <= 50)]

# 初始化空的训练集和验证集
train_df = pd.DataFrame()
val_df = pd.DataFrame()

# 按长度分组
for length, group in df_filtered.groupby('length'):
    # 检查样本数是否足够
    if len(group) >= 50000:
        # 随机划分 40,000 训练样本和 10,000 验证样本
        train_group, val_group = train_test_split(group, train_size=40000, test_size=10000, random_state=42)
        train_df = pd.concat([train_df, train_group])
        val_df = pd.concat([val_df, val_group])
    else:
        print(f"长度 {length} 的样本数不足 50,000，已跳过")

# 删除长度列，因为原始数据不需要这个列
train_df = train_df.drop(columns=['length'])
val_df = val_df.drop(columns=['length'])

# 保存训练集和验证集到新的CSV文件
train_df.to_csv('pretrain_data.csv', index=False)
val_df.to_csv('preval_data.csv', index=False)

# 统计训练集和验证集中每个长度的样本数
train_length_counts = train_df['Sequence'].apply(len).value_counts().sort_index()
val_length_counts = val_df['Sequence'].apply(len).value_counts().sort_index()

# 输出结果
print("训练集中每个长度的样本数：")
print(train_length_counts)

print("\n验证集中每个长度的样本数：")
print(val_length_counts)

print("\n训练集和验证集已成功拆分并保存！")
import pandas as pd

input_csv = 'grampa.csv'
output_fasta = 'posSaureus.fasta'

# 列索引配置
species_column = 1  # 第二列（E. coli筛选）
sequence_column = 3  # 第四列（序列）
value_column = 7  # 第八列（数值排序）

# 读取CSV文件（无标题则设置 header=None）
df = pd.read_csv(input_csv, header=None)

# 数据预处理
df = (
    df
    # 筛选第二列为"E. coli"的行
    .loc[df[species_column] == 'S. aureus']
    # 提取序列并计算长度
    .assign(
        sequence=lambda x: x[sequence_column].str.strip(),
        length=lambda x: x['sequence'].str.len()
    )
    # 过滤长度≤50的序列
    .loc[lambda x: (x['length'] <= 50) & (x['length'] >= 5)]
    # 去重（保留首次出现的唯一序列）
    .drop_duplicates(subset=['sequence'])
    # 转换第八列为数值类型（无法转换的设为NaN）
    .assign(
        value=lambda x: pd.to_numeric(x[value_column], errors='coerce')
    )
    # 删除第八列无效的行（NaN或非数值）
    .dropna(subset=['value'])
    # 仅保留 value < 0 的行
    #.loc[lambda x: x['value'] >= 1.305]
    .loc[lambda x: x['value'] <= 1]
    # 按第八列升序排序
    .sort_values(by='value')

)

# 写入FASTA文件
with open(output_fasta, 'w') as f:
    for i, (index, row) in enumerate(df.iterrows(), start=1):
        sequence = row['sequence']
        f.write(f'>seq{i}\n{sequence}\n')
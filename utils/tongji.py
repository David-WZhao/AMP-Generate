import pandas as pd


def count_bacteria(csv_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 假设第二列是细菌名字，选择第二列
    bacteria_column = df.iloc[:, 1]

    # 统计不同细菌及其出现的次数
    bacteria_counts = bacteria_column.value_counts().reset_index()

    # 将列名重命名为 "Bacteria" 和 "Count"
    bacteria_counts.columns = ['Bacteria', 'Count']

    # 输出结果到CSV文件
    bacteria_counts.to_csv(output_file, index=False)
    print(f"统计结果已保存到 {output_file}")


# 调用函数，输入CSV文件路径和输出文件路径
csv_file = 'grampa.csv'  # 替换为实际的CSV文件路径
output_file = 'bacteria_counts.csv'  # 输出文件名
count_bacteria(csv_file, output_file)

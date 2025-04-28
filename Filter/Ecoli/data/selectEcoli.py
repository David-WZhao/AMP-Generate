import pandas as pd


def filter_e_coli(input_file, output_file):
    # 读取 CSV 文件
    df = pd.read_csv(input_file, header=None)  # 假设没有列名

    # 筛选第二列值为 'E. coli' 的行
    filtered_df = df[df[1] == 'E. coli']

    # 只保留第四列和第八列
    filtered_df = filtered_df.iloc[:, [3, 7]]

    # 去重
    filtered_df = filtered_df.drop_duplicates()

    # 保存筛选后的数据到新文件
    filtered_df.to_csv(output_file, index=False, header=False)


# 示例用法
input_file = "grampa.csv"  # 替换为实际文件路径
output_file = "Ecoli.csv"
filter_e_coli(input_file, output_file)
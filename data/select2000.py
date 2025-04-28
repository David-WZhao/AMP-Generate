import pandas as pd

def sample_csv_data(file_path, output_path, sample_size=2000):
    """
    从 CSV 文件中随机抽取指定数量的数据，并保存到新的 CSV 文件。

    参数:
    - file_path: 输入的 CSV 文件路径
    - output_path: 输出的 CSV 文件路径
    - sample_size: 需要抽取的数据条数，默认 2000 条
    """
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 确保数据量足够，否则取全部数据
    sample_size = min(sample_size, len(df))

    # 随机抽取数据
    sampled_df = df.sample(n=sample_size, random_state=40)  # 设置 random_state 以便复现

    # 保存到新的 CSV 文件
    sampled_df.to_csv(output_path, index=False)

    print(f"成功抽取 {sample_size} 条数据，已保存至 {output_path}")

# 示例调用
input_csv = "neg_data.csv"  # 替换为你的 CSV 文件路径
output_csv = "neg_data_Saureus.csv"  # 采样后的数据文件
sample_csv_data(input_csv, output_csv)

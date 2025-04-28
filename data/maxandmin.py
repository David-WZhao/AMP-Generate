import pandas as pd

def calculate_min_max(file_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 提取从第三列到最后一列的数据
    data = df.iloc[:, 2:]

    # 计算每列的最大值和最小值
    min_values = data.min()
    max_values = data.max()

    # 输出结果
    print("每列的最小值：")
    print(min_values)

    print("\n每列的最大值：")
    print(max_values)


# 使用示例
file_path = "pre_train_clean.csv"  # 替换为你的文件路径
calculate_min_max(file_path)
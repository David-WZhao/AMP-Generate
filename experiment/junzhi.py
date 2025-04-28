import pandas as pd


def calculate_column_averages(file_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 确保至少有三列数据
    if df.shape[1] < 3:
        print("CSV 文件至少需要三列数据！")
        return

    # 计算第三列到最后一列的平均值
    column_means = df.iloc[:, 2:].mean()

    # 打印平均值
    print("每列的平均值:")
    print(column_means)

    return column_means


# 示例使用
file_path = "posSaureus_guiyi.csv"  # 替换为你的 CSV 文件路径
calculate_column_averages(file_path)

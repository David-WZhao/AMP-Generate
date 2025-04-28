import pandas as pd
import numpy as np

def normalize_matrix(matrix, min_values, max_values):
    """
    归一化矩阵，使其归一化到 [0, 1] 范围。

    参数:
    - matrix: 需要归一化的 numpy 数组
    - min_values: 每列的最小值，形状与 matrix 列数匹配
    - max_values: 每列的最大值，形状与 matrix 列数匹配

    返回:
    - 归一化后的 numpy 数组
    """
    return (matrix - min_values) / (max_values - min_values)

def normalize_csv(file_path, output_path):
    """
    读取 CSV 文件，对第三列到最后一列的数据进行归一化，并保存到新的 CSV 文件。

    参数:
    - file_path: 输入的 CSV 文件路径
    - output_path: 输出的 CSV 文件路径
    """
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 提取需要归一化的列（第三列到最后一列）
    cols_to_normalize = df.columns[2:]

    # 归一化的最大最小值（对应第三列到最后一列的属性）
    min_values = np.array([75.0666, 4.05, -4.5, 0.0, -112.24, 0.0, 60.1, 0.0, 0.0, 0.0])
    max_values = np.array([6908.82, 12.0, 4.5, 1.0, 485.94, 66000.0, 8185.2, 1.0, 1.0, 1.0])

    # 取出需要归一化的数据并转换为 numpy 数组
    matrix = df[cols_to_normalize].to_numpy()

    # 归一化
    normalized_matrix = normalize_matrix(matrix, min_values, max_values)

    # 替换原数据
    df[cols_to_normalize] = normalized_matrix

    # 保存归一化后的数据到新文件
    df.to_csv(output_path, index=False)

# 示例调用
input_csv = "pos_geneate_Saureus_Cecropin_gai.csv"  # 你的输入文件路径
output_csv = "pos_geneate_Saureus_Cecropin_gai_guiyi.csv"  # 你的输出文件路径
normalize_csv(input_csv, output_csv)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_columns_scaler(input_files, output_files):
    # 读取所有文件
    dfs = [pd.read_csv(file) for file in input_files]

    # 提取需要归一化的列（从第 3 列开始）
    data_to_scale = pd.concat([df.iloc[:, 2:] for df in dfs], axis=0)

    # 创建并应用缩放器
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_to_scale)

    # 将归一化后的数据拆分回各自的文件
    start_idx = 0
    for i, df in enumerate(dfs):
        # 获取当前文件的归一化列数
        num_cols = df.shape[1] - 2
        # 提取当前文件的归一化数据
        scaled_df = pd.DataFrame(
            scaled_data[start_idx:start_idx + len(df), :],
            columns=df.columns[2:],
            index=df.index
        )
        # 更新起始索引
        start_idx += len(df)
        # 合并前两列和归一化后的列
        final_df = pd.concat([df.iloc[:, :2], scaled_df], axis=1)
        # 保存结果
        final_df.to_csv(output_files[i], index=False)
        print(f"处理完成，文件已保存至: {output_files[i]}")

    # 检查常数列
    constant_cols = data_to_scale.columns[data_to_scale.nunique() == 1]
    for col in constant_cols:
        print(f"提示: 列 '{col}' 为常数列，已自动标准化为0")

    # 输出每个属性的最大值
    max_values = pd.DataFrame(scaled_data, columns=data_to_scale.columns).max()
    print("\n每个属性的最大值：")
    print(max_values)


# 使用示例
input_files = [
    "pretrain_data.csv",
    "preval_data.csv",
    "LatentDiffusion_Train.csv",
    "LatentDiffusion_Val.csv",
    "pos_data.csv",
    "neg_data.csv"
]
output_files = [
    "pretrain_data.csv",
    "preval_data.csv",
    "LatentDiffusion_Train.csv",
    "LatentDiffusion_Val.csv",
    "pos_data.csv",
    "neg_data.csv"
]
normalize_columns_scaler(input_files, output_files)
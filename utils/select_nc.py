import pandas as pd


def csv_to_fasta(csv_file, fasta_file):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 确保 CSV 至少有四列
    if df.shape[1] < 4:
        raise ValueError("CSV 文件至少需要四列")

    # 过滤掉第三列值大于 0.1 的行
    third_col_name = df.columns[2]
    df_filtered = df[df[third_col_name] <= 0.1].reset_index(drop=True)

    # 计算序列长度
    df_filtered["Seq_Length"] = df_filtered.iloc[:, 3].apply(len)

    # 仅保留序列长度小于等于50的
    df_filtered = df_filtered[df_filtered["Seq_Length"] <= 50]

    # 按序列长度分组，并在每个长度内选择第二列值最小的前100个
    second_col_name = df_filtered.columns[1]
    df_selected = df_filtered.sort_values(by=["Seq_Length", second_col_name]).groupby("Seq_Length").head(100)

    # 生成 FASTA 序列
    fasta_sequences = [f">seq{i + 1}\n{row[3]}" for i, row in df_selected.iterrows()]

    # 将结果保存为 FASTA 文件
    with open(fasta_file, "w") as f:
        f.write("\n".join(fasta_sequences))

    print(f"FASTA 文件已保存为 {fasta_file}")


# 示例用法
csv_to_fasta("nonamppredict.csv", "nonamppredict.fasta")
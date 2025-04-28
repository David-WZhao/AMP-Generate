import pandas as pd

def filter_csv_by_fasta(csv_file, fasta_file, output_csv):
    # 读取 FASTA 文件中的所有序列
    fasta_sequences = set()
    with open(fasta_file, "r") as f:
        lines = f.readlines()
        for i in range(1, len(lines), 2):  # 仅获取序列行
            fasta_sequences.add(lines[i].strip())

    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 假设第二列是序列内容（调整列索引）
    sequence_col = df.columns[1]  # 确保是正确的列

    # 过滤出 CSV 中第二列在 FASTA 文件中的行
    df_filtered = df[df[sequence_col].isin(fasta_sequences)]

    # 保存结果为新的 CSV 文件
    df_filtered.to_csv(output_csv, index=False)

    print(f"筛选后的 CSV 文件已保存为 {output_csv}")

# 示例用法
filter_csv_by_fasta("neg_data.csv", "nonamppredict.fasta", "neg_data_gai.csv")
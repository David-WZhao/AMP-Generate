import pandas as pd


def csv_to_fasta(csv_file, fasta_file):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 确保 CSV 至少有四列
    if df.shape[1] < 4:
        raise ValueError("CSV 文件至少需要四列")

    # 假设第二列的名称是 'Category'，如果不同请修改
    second_col_name = df.columns[1]

    # 过滤出第二列为 "Non-AMP" 的行
    filtered_df = df[df[second_col_name] == "AMP"].reset_index(drop=True)

    # 假设第四列是序列内容
    fasta_sequences = []
    for i, row in filtered_df.iterrows():
        fasta_sequences.append(f">seq{i + 1}\n{row[3]}")

    # 将结果保存为 FASTA 文件
    with open(fasta_file, "w") as f:
        f.write("\n".join(fasta_sequences))

    print(f"FASTA 文件已保存为 {fasta_file}")


# 示例用法
csv_to_fasta("Ecoli_shaixuan.csv", "Ecoli.fasta")

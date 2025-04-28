def txt_to_fasta(txt_file, fasta_file):
    # 读取文件中的所有行
    with open(txt_file, "r") as f:
        sequences = [line.strip() for line in f.readlines() if line.strip()]

    # 生成 FASTA 格式的序列
    fasta_sequences = [f">seq{i+1}\n{seq}" for i, seq in enumerate(sequences)]

    # 将 FASTA 序列写入文件
    with open(fasta_file, "w") as f:
        f.write("\n".join(fasta_sequences))

    print(f"FASTA 文件已保存为 {fasta_file}")

# 示例用法
txt_to_fasta("DeepAMP", "DeepAMP.fasta")

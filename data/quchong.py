import csv

input_csv = "pre_train.csv"      # 原始的CSV文件
dedup_fasta = "pre_train_clean.fasta"  # 去重后的FASTA文件

seen = set()

with open(input_csv, 'r', newline='', encoding='utf-8') as infile, \
     open(dedup_fasta, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    header = next(reader, None)  # 读取表头并跳过

    for row in reader:
        if len(row) < 2:
            # 如果行数据不完整，跳过或根据实际需求处理
            continue
        seq_id = row[0]  # 第一列为ID
        seq = row[1]     # 第二列为序列
        if seq not in seen:
            # 将该序列以FASTA格式写入输出文件
            outfile.write(f">{seq_id}\n")
            outfile.write(f"{seq}\n")
            seen.add(seq)

print("去重完成。结果已写入", dedup_fasta)

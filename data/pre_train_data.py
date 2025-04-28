import csv
import numpy as np
from collections import defaultdict

fasta_file = "uniprotkb_reviewed_true_2024_12_17.fasta"  # 请替换为您的FASTA文件名
output_csv = "pre_train.csv"  # 输出的CSV文件名

def read_fasta(filepath):
    sequences = []
    with open(filepath, 'r') as f:
        seq_id = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # 如果之前有积累的序列，则加入列表
                if seq_id is not None and seq_lines:
                    sequences.append("".join(seq_lines))
                seq_id = line[1:]  # 去掉 '>'
                seq_lines = []
            else:
                seq_lines.append(line)
        # 文件末尾可能还有最后一个序列需要存储
        if seq_id is not None and seq_lines:
            sequences.append("".join(seq_lines))
    return sequences

all_sequences = read_fasta(fasta_file)

# 定义裁剪长度范围
min_len = 2
max_len = 50

# 统计每个长度的目标数量
total_segments = len(all_sequences) * 10  # 假设每个序列平均裁剪10个片段
target_count_per_length = total_segments // (max_len - min_len + 1)

# 用于记录每个长度的实际裁剪数量
length_counts = defaultdict(int)

with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # 写表头（可选）
    writer.writerow(["ID", "Truncated_Sequence"])

    current_id = 1
    for seq in all_sequences:
        start = 0
        seq_length = len(seq)

        # 当还有剩余序列需要截断
        while start < seq_length:
            # 随机选择一个长度，优先选择还未达到目标数量的长度
            available_lengths = [l for l in range(min_len, max_len + 1) if length_counts[l] < target_count_per_length]
            if not available_lengths:
                break  # 所有长度都已达到目标数量

            segment_len = np.random.choice(available_lengths)

            # 若超过剩余序列长度则设为剩余长度
            remaining = seq_length - start
            if segment_len > remaining:
                segment_len = remaining

            truncated_seq = seq[start:start + segment_len]
            writer.writerow([current_id, truncated_seq])

            length_counts[segment_len] += 1
            current_id += 1
            start += segment_len  # 更新起始位置，继续处理下一个片段

# 输出每个长度最终的裁剪个数
print("每个长度最终的裁剪个数：")
for length in range(min_len, max_len + 1):
    print(f"长度 {length}: {length_counts[length]} 个")

print("处理完成，结果已写入", output_csv)
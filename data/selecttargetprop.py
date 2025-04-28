import csv


def analyze_csv(file_path):
    # 读取CSV文件并提取数据（假设无标题行）
    rows = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                # 提取第三列到最后一列的浮点数值
                cols = list(map(float, row[2:]))
                rows.append({'raw': row, 'numeric': cols})
            except ValueError:
                continue  # 跳过无效行

    # 定义需较大差异的列索引（从第三列开始为索引0）
    target_indices = [2, 3, 5, 7]  # 对应原第五、六、八、十列
    best_pair = None

    # 遍历所有行对
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            row1 = rows[i]['numeric']
            row2 = rows[j]['numeric']

            # 检查目标列差异是否均≥0.25
            target_diff_ok = all(
                abs(row1[col] - row2[col]) >= 0.22
                for col in target_indices
            )

            # 检查其他列差异是否均≤0.05
            other_cols = [
                col for col in range(len(row1))
                if col not in target_indices
            ]
            other_diff_ok = all(
                abs(row1[col] - row2[col]) <= 0.07
                for col in other_cols
            )

            if target_diff_ok and other_diff_ok:
                best_pair = (rows[i]['raw'], rows[j]['raw'])
                return best_pair  # 返回首个匹配对

    return best_pair


# 示例调用
result = analyze_csv('posSaureus_guiyi.csv')
if result:
    print("满足条件的行组：")
    print("行1：", result[0])
    print("行2：", result[1])
else:
    print("未找到符合条件的行组")
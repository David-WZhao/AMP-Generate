import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import EsmTokenizer
import torch.nn as nn
from main import *
# 读取 DNA FASTA

def read_fasta(file_path):
    sequences = []
    with open(file_path, "r") as file:
        sequence = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                sequence = ""
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    return sequences


# 预测函数
def predict_mic(fasta_file, model_path, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESMRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sequences = read_fasta(fasta_file)
    dna_kmers = prepare_dna_data()
    dna_kmers = [dna_kmers] * len(sequences)

    dataset = MICDataset(sequences, dna_kmers, [0] * len(sequences))  # 伪标签
    data_loader = DataLoader(dataset, batch_size=CFG['batch_size'], collate_fn=collate_fn)

    predictions = []
    with torch.no_grad():
        for seqs, kmers, _ in data_loader:
            preds = model(seqs, kmers.to(device)).cpu().numpy()

            # 确保 preds 是一维数组
            if preds.ndim == 0:
                preds = [preds.item()]  # 处理单个标量输出
            else:
                preds = preds.tolist()  # 转换为 Python 列表

            predictions.extend(preds)

    df = pd.DataFrame({"Sequence": sequences, "Predicted_MIC": predictions})
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


# 示例调用
if __name__ == "__main__":
    fasta_input = "DeepAMP.fasta"  # 你的输入FASTA文件
    model_checkpoint = "best_model.pth"  # 训练好的模型权重
    output_file = "DeepAMP_ecoli.csv"  # 预测结果输出文件
    predict_mic(fasta_input, model_checkpoint, output_file)
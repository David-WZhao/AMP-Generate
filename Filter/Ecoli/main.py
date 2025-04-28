import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import EsmTokenizer, EsmModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from scipy.stats import spearmanr, pearsonr
from collections import Counter
import itertools

# 配置参数
CFG = {
    'esm_dir': "./esm",  # ESM 预训练模型路径
    'batch_size': 8,
    'esm_dim': 1280,
    'hidden_dim': 128,
    'dropout': 0.5,
    'lr': 1e-4,
    'epochs': 100,
    'patience': 10,
    'max_length': 1024,
    'weight_decay': 1e-5,
    'dna_fasta': "data/sequence.fasta",  # DNA FASTA 文件路径
    'csv_file': "data/Ecoli.csv"  # MIC 数据 CSV 文件路径
}


# 读取 MIC 数据
def prepare_data():
    df = pd.read_csv(CFG['csv_file'], header=None, names=["Sequence", "MIC"])
    df = df.dropna(subset=['Sequence', 'MIC'])
    df['MIC'] = pd.to_numeric(df['MIC'], errors='coerce')
    df = df.dropna(subset=['MIC'])
    return df['Sequence'].tolist(), df['MIC'].values


# 读取 DNA FASTA
def read_fasta(file_path):
    sequences = {}
    with open(file_path, "r") as file:
        sequence_id = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence_id:
                    sequences[sequence_id] = "".join(sequence)
                sequence_id = line[1:]
                sequence = []
            else:
                sequence.append(line)
        if sequence_id:
            sequences[sequence_id] = "".join(sequence)
    return sequences


# 计算 k-mer 频率
def kmer_frequency(sequence, k=6):
    bases = ['A', 'C', 'G', 'T']
    all_kmers = ["".join(p) for p in itertools.product(bases, repeat=k)]
    kmer_counts = Counter([sequence[i:i + k] for i in range(len(sequence) - k + 1)])
    total_kmers = sum(kmer_counts.values())
    return [kmer_counts.get(kmer, 0) / total_kmers for kmer in all_kmers]


# 读取 DNA 并计算 k-mer 特征
def prepare_dna_data():
    dna_sequences = read_fasta(CFG['dna_fasta'])
    full_dna_sequence = "".join(dna_sequences.values())  # 合并所有 DNA 片段
    return kmer_frequency(full_dna_sequence)


# 定义 ESM 模型包装类
class ESMWrapper:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EsmModel.from_pretrained(CFG['esm_dir']).to(self.device)
        self.tokenizer = EsmTokenizer.from_pretrained(CFG['esm_dir'])
        for param in self.model.parameters():
            param.requires_grad = False

    def get_embeddings(self, sequences):
        inputs = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=CFG['max_length'],
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.cpu()


# 注意力池化
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, hidden_states):
        attn_weights = self.attn(hidden_states)
        return torch.sum(attn_weights * hidden_states, dim=1)


# 定义 ESM 回归模型
class ESMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 添加这一行
        self.esm = ESMWrapper()
        self.pool = AttentionPooling(CFG['esm_dim'])

        self.dna_fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.regressor = nn.Sequential(
            nn.Linear(CFG['esm_dim'] + 256, CFG['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(CFG['dropout']),
            nn.Linear(CFG['hidden_dim'], 1)
        )

    def forward(self, seqs, dna_kmers):
        embeddings = self.esm.get_embeddings(seqs).to(self.device)  # 确保使用 self.device
        pooled = self.pool(embeddings)
        dna_features = self.dna_fc(dna_kmers.to(self.device))
        combined_features = torch.cat([pooled, dna_features], dim=1)
        return self.regressor(combined_features).squeeze()


# 数据集类
class MICDataset(Dataset):
    def __init__(self, sequences, dna_kmers, labels):
        self.sequences = sequences
        self.dna_kmers = torch.FloatTensor(dna_kmers)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.dna_kmers[idx], self.labels[idx]


def collate_fn(batch):
    seqs, dna_kmers, labels = zip(*batch)
    return list(seqs), torch.stack(dna_kmers), torch.stack(labels)


def evaluate_model(model, test_loader, device):
    model.eval()
    val_pred, val_true = [], []
    with torch.no_grad():
        for seqs, kmers, labels in test_loader:
            labels = labels.to(device)  # 确保 labels 在正确的设备上
            preds = model(seqs, kmers).cpu()  # 预测结果转到 CPU
            val_pred.extend(preds.numpy())
            val_true.extend(labels.cpu().numpy())  # 先移动到 CPU，再转 NumPy

    mse = mean_squared_error(val_true, val_pred)
    mae = mean_absolute_error(val_true, val_pred)
    r2 = r2_score(val_true, val_pred)
    pearson_corr, _ = pearsonr(val_true, val_pred)
    spearman_corr, _ = spearmanr(val_true, val_pred)

    print(f"\\nEvaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")
    return mse

# 训练模型
def train_model():
    sequences, y = prepare_data()
    dna_kmers = prepare_dna_data()
    dna_kmers = [dna_kmers] * len(sequences)

    X_train, X_test, y_train, y_test, kmers_train, kmers_test = train_test_split(sequences, y, dna_kmers, test_size=0.2,
                                                                                 random_state=42)

    train_loader = DataLoader(MICDataset(X_train, kmers_train, y_train), batch_size=CFG['batch_size'], shuffle=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(MICDataset(X_test, kmers_test, y_test), batch_size=CFG['batch_size'],
                             collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESMRegressor().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    criterion = nn.MSELoss()

    best_loss, patience = float('inf'), 0
    for epoch in range(CFG['epochs']):
        model.train()
        total_loss = 0
        for seqs, kmers, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs, kmers)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        print(f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader.dataset):.4f}")
        val_loss = evaluate_model(model, test_loader, device)
        best_model_path = "best_model.pth"
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path}")
        else:
            patience += 1
            if patience >= CFG['patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break

if __name__ == "__main__":
    train_model()
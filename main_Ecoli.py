import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.utils.data import random_split
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset, DataLoader


class MoleculeDataset(Dataset):
    def __init__(self, sequences, properties):
        self.sequences = sequences
        self.properties = properties

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 确保将 numpy 数组转换为 Tensor 后再使用 clone().detach()
        sequence = self.sequences[idx]
        properties = torch.tensor(self.properties[idx], dtype=torch.float32)

        # 现在可以使用 clone().detach() 避免警告
        return sequence.clone().detach(), properties.clone().detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True, type=str,
                        help=
                        "TransVAE,Reconstruct_test,\
                        GetMem_nc,\
                        GetMem_c,\
                        LatentDiffusion_nocondition,\
                        LatentDiffusion_condition,\
                        Generate"
                        )

    parser.add_argument("--vae_epoch", default=200, type=int)
    parser.add_argument("--vae_batch_size", default=512, type=int)
    parser.add_argument("--vae_lr", default=0.0007, type=float)
    parser.add_argument("--vae_save_path", default="./model_mulgpu", type=str)
    parser.add_argument("--vae_train_path", default="./data/pretrain_data.csv", type=str)
    parser.add_argument("--vae_val_path", default="./data/preval_data.csv", type=str)
    parser.add_argument("--vae_model_path",
                        default="./model_mulgpu/model/model_027_0.09154441227554247_1.8960851255663254_.pth", type=str)

    parser.add_argument("--mem_save_path_nc", default="./memory", type=str)
    parser.add_argument("--mem_save_path_c", default="./memory_c", type=str)

    parser.add_argument("--LatentDiffusion_lr", default=0.0001, type=float)
    parser.add_argument("--LatentDiffusion_save_path_nc", default="./model_mulgpu_nc", type=str)
    parser.add_argument("--LatentDiffusion_epoch", default=200, type=int)
    parser.add_argument("--LatentDiffusion_batch_size", default=512, type=int)
    parser.add_argument("--LatentDiffusion_num_steps", default=500, type=int)
    parser.add_argument("--LatentDiffusion_shuffle", default=False, type=bool)
    parser.add_argument("--LatentDiffusion_num_workers", default=8, type=int)
    parser.add_argument("--LatentDiffusion_pin_memory", default=True, type=bool)
    parser.add_argument("--LatentDiffusion_drop_last", default=True, type=bool)
    parser.add_argument("--LatentDiffusion_save_path_c", default="./model_mulgpu_c_ecoli", type=str)
    parser.add_argument("--LatentDiffusion_epoch_c", default=200, type=int)

    parser.add_argument("--Generate_VAE_model_path",
                        default="./model_mulgpu/model/model_027_0.09154441227554247_1.8960851255663254_.pth"
                                "", type=str)
    parser.add_argument("--Generate_times", default=1, type=int)
    parser.add_argument("--Generate_batch_num", default=512, type=int)
    parser.add_argument("--Generate_batch_times", default=1, type=int)
    parser.add_argument("--Generate_condition", default=1, type=int)
    parser.add_argument("--Generate_Diffusion_model_path", default="./model_mulgpu_c_ecoli/best_model.pth", type=str)
    parser.add_argument("--Generate_save_path", default="./pos_geneate_ecoli.fasta", type=str)
    parser.add_argument("--Generate_tem_path", default="./diffusion_data_tem", type=str)
    parser.add_argument("--Generate_mem_path", default="./diffusion_data_mem", type=str)
    args = parser.parse_args()

    if args.work == "TransVAE":
        from src.TransVAE import *

        # 移除分布式训练相关设置
        # torch.backends.cudnn.benchmark = True   # 如果你不再使用分布式训练，这一行可以去掉

        # 选择设备（CPU 或者单个 GPU）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 训练参数
        train_params = {
            'BATCH_SIZE': args.vae_batch_size,
            'BATCH_CHUNKS': 1,
            "Save_Path": args.vae_save_path,
            "BETA_INIT": 1e-8,
            "BETA": 0.05,
            "ANNEAL_START": 0,
            "Epochs": args.vae_epoch,
            "LR_SCALE": 1,
            "WARMUP_STEPS": 10000,
        }

        # 加载数据
        # train_mols = pd.read_csv("./data/VAE_Train").to_numpy()
        # print(train_mols[0])  ['MLLPDNNMLGYSQYFKIIVDTKDKLNSSLEIENIDY']
        # print(type(train_mols))  numpy
        # print(type(train_mols[0]))  numpy
        # print(type(train_mols[0][0])) str
        # print(train_mols[0][0])MLLPDNNMLGYSQYFKIIVDTKDKLNSSLEIENIDY
        # val_mols = pd.read_csv("VAE_Val").to_numpy()
        # train_data = vae_data_gen(train_mols, params["src_len"], char_dict=w2i)
        # val_data = vae_data_gen(val_mols, params["src_len"], char_dict=w2i)
        # print(train_data[0])    用w2i编码  得到数字序列 开始是0  结尾是0   vae_data_gen的作用
        # print(w2i) {'<start>': 0, 'E': 1, 'I': 2, 'F': 3, 'A': 4, 'K': 5, 'N': 6, 'G': 7, 'M': 8, 'T': 9, 'P': 10, 'L': 11, 'R': 12, 'D': 13, 'S': 14, 'V': 15, 'Y': 16, 'Q': 17, 'C': 18, 'W': 19, 'H': 20, '_': 21, '<end>': 22}

        train_mols = pd.read_csv(args.vae_train_path)['Sequence'].to_numpy().reshape(-1, 1)
        # train_mols_properties = pd.read_csv(args.vae_train_path, nrows = 2000000)
        #
        # # print(type(train_mols))  #numpy
        # # print((train_mols[0]))  #numpy
        # # print((train_mols[0][0])) #str
        #
        val_mols = pd.read_csv(args.vae_val_path)['Sequence'].to_numpy().reshape(-1, 1)
        #
        #
        # 对序列进行编码
        train_data = vae_data_gen(train_mols, params["src_len"], char_dict=w2i)
        val_data = vae_data_gen(val_mols, params["src_len"], char_dict=w2i)

        # 引入性质数据
        train_mols_properties = pd.read_csv(args.vae_train_path)
        val_mols_properties = pd.read_csv(args.vae_train_path)
        # 选择第三列到最后一列的十个值（假设是第3到第12列，索引2到11）
        train_mols_properties_values = train_mols_properties.iloc[:, 2:12].to_numpy()
        val_mols_properties_values = train_mols_properties.iloc[:, 2:12].to_numpy()
        # print(train_mols_properties_values[0])
        # 创建一个 MinMaxScaler 对象
        # scaler = MinMaxScaler()
        #
        # # 对数据进行归一化
        # train_mols_properties_values_normalized = scaler.fit_transform(train_mols_properties_values)
        # val_mols_properties_values_normalized = scaler.fit_transform(train_mols_properties_values)
        # print(train_mols_properties_values_normalized[0])
        # 创建模型
        train_dataset = MoleculeDataset(train_data, train_mols_properties_values)
        val_dataset = MoleculeDataset(val_data, val_mols_properties_values)

        model = create_VAE()
        model.to(device)

        # DataLoader 不再需要分布式采样器
        train_iter = torch.utils.data.DataLoader(
            # train_data,
            train_dataset,
            batch_size=train_params['BATCH_SIZE'],
            shuffle=True,  # 使用常规的shuffle
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )

        val_iter = torch.utils.data.DataLoader(
            # val_data,
            val_dataset,
            batch_size=train_params['BATCH_SIZE'],
            shuffle=False,  # 验证时不需要shuffle
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )

        # 保存路径设置
        if not os.path.exists(f"{train_params['Save_Path']}"):
            os.makedirs(f"{train_params['Save_Path']}")
        if not os.path.exists(f"{train_params['Save_Path']}/model"):
            os.makedirs(f"{train_params['Save_Path']}/model")

        log_filepath = f"{train_params['Save_Path']}/train.log"
        try:
            f = open(log_filepath, 'r')
            f.close()
            already_wrote = True
        except FileNotFoundError:
            already_wrote = False
        log_file = open(log_filepath, 'a')
        if not already_wrote:
            log_file.write('epoch,batch_idx,data_type,tot_loss,recon_loss,pred_loss,run_time\n')
        log_file.close()

        # KLAnnealer 和优化器
        kl_annealer = KLAnnealer(train_params['BETA_INIT'], train_params['BETA'], train_params['Epochs'],
                                 train_params['ANNEAL_START'])
        optimizer = NoamOpt(
            params['d_model'], train_params['LR_SCALE'], train_params['WARMUP_STEPS'],
            torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9),
            args.vae_lr
        )

        CHAR_WEIGHTS = torch.tensor(char_weights, dtype=torch.float).to(device)

        print("Run train.")
        for epoch in range(train_params['Epochs']):
            epoch_start_time = time.time()
            model.train()
            losses = []
            beta = kl_annealer(epoch)

            # 训练阶段
            for j, batch in enumerate(train_iter):
                # print(data[0])  对每个序列的氨基酸进行了编码
                data, property = batch
                # print(data.shape,property.shape)
                property = property.clone().detach().to(torch.float32).to(device)

                # print(j)
                avg_losses = []
                avg_bce_losses = []
                avg_bcemask_losses = []
                avg_kld_losses = []
                avg_pp_losses = []
                start_run_time = time.time()
                mols_data = data[:, :-1]
                # print(mols_data.shape)   512  127
                mols_data = mols_data.to(device)
                src = Variable(mols_data).long()
                # print(src[0])  512 127
                tgt = Variable(mols_data[:, :-1]).long()
                # print(tgt[0])  512 126
                src_mask = (src != w2i["_"]).unsqueeze(-2)
                tgt_mask = make_std_mask(tgt, w2i["_"])
                # add pp mask
                # property_mask = torch.ones_like(property, dtype=torch.bool)
                pp_src = property.clone().detach().to(torch.float32).to(device)
                # pp_tgt = property.unsqueeze(1).expand(-1, tgt.size(1), -1).clone().detach().to(torch.float32).to(device)

                # print(src_mask[0],property_mask[0])   [True False]
                # 待增加pp_src_mask  pp_tgt_mask
                # print(a)
                x_out, mu, logvar, pred_len, pred_pp = model(src, tgt, pp_src, src_mask, tgt_mask)
                true_len = src_mask.sum(dim=-1)
                #   transvae  过程中的维度转化
                loss, bce, bce_mask, kld, pp_loss = trans_vae_loss(src, x_out, mu, logvar,
                                                                   true_len, pred_len,
                                                                   CHAR_WEIGHTS, property, pred_pp, beta)
                avg_bcemask_losses.append(bce_mask.item())
                avg_losses.append(loss.item())
                avg_bce_losses.append(bce.item())
                avg_kld_losses.append(kld.item())
                avg_pp_losses.append(pp_loss.item())
                loss.backward()
                optimizer.step()
                model.zero_grad()
                stop_run_time = time.time()
                run_time = round(stop_run_time - start_run_time, 5)
                avg_loss = np.mean(avg_losses)
                avg_bce = np.mean(avg_bce_losses)
                avg_bcemask = np.mean(avg_bcemask_losses) if avg_bcemask_losses else 0
                avg_kld = np.mean(avg_kld_losses)
                avg_pp = np.mean(avg_pp_losses)
                losses.append(avg_loss)

                log_file = open(log_filepath, 'a')
                log_file.write('{},{},{},{},{},{},{},{},{}\n'.format(
                    epoch,
                    j, 'train',
                    avg_loss,
                    avg_bce,
                    avg_bcemask,
                    avg_kld,
                    avg_pp,
                    run_time))
                log_file.close()

            train_loss = np.mean(losses)
            train_time = time.time() - epoch_start_time
            val_start_time = time.time()
            # 验证阶段
            model.eval()
            val_losses = []
            for j, batch in enumerate(val_iter):
                data, property = batch
                property = property.clone().detach().to(torch.float32).to(device)

                avg_losses = []
                avg_bce_losses = []
                avg_bcemask_losses = []
                avg_kld_losses = []
                avg_pp_losses = []
                start_run_time = time.time()
                mols_data = data[:, :-1]
                mols_data = mols_data.to(device)
                src = Variable(mols_data).long()
                tgt = Variable(mols_data[:, :-1]).long()
                src_mask = (src != w2i["_"]).unsqueeze(-2)
                tgt_mask = make_std_mask(tgt, w2i["_"])
                pp_src = property.clone().detach().to(torch.float32).to(device)
                # pp_tgt = property.unsqueeze(1).expand(-1, tgt.size(1), -1).clone().detach().to(torch.float32).to(device)
                scores = Variable(data[:, -1])
                x_out, mu, logvar, pred_len, pred_pp = model(src, tgt, pp_src, src_mask, tgt_mask)
                true_len = src_mask.sum(dim=-1)
                #   transvae  过程中的维度转化
                loss, bce, bce_mask, kld, pp_loss = trans_vae_loss(src, x_out, mu, logvar,
                                                                   true_len, pred_len,
                                                                   CHAR_WEIGHTS, property, pred_pp, beta)
                avg_bcemask_losses.append(bce_mask.item())
                avg_losses.append(loss.item())
                avg_bce_losses.append(bce.item())
                avg_kld_losses.append(kld.item())
                avg_pp_losses.append(pp_loss.item())
                stop_run_time = time.time()
                run_time = round(stop_run_time - start_run_time, 5)
                avg_loss = np.mean(avg_losses)
                avg_bce = np.mean(avg_bce_losses)
                avg_bcemask = np.mean(avg_bcemask_losses) if avg_bcemask_losses else 0
                avg_kld = np.mean(avg_kld_losses)
                avg_pp = np.mean(avg_pp_losses)
                val_losses.append(avg_loss)

                log_file = open(log_filepath, 'a')
                log_file.write('{},{},{},{},{},{},{},{},{}\n'.format(
                    epoch,
                    j, 'test',
                    avg_loss,
                    avg_bce,
                    avg_bcemask,
                    avg_kld,
                    avg_pp,
                    run_time))
                log_file.close()

            val_loss = np.mean(val_losses)
            epoch_end_time = time.time()
            val_time = round(epoch_end_time - val_start_time, 5)

            print(
                f'Epoch - {epoch} Train - {train_loss} Val - {val_loss} KLBeta - {beta} Epoch time - {train_time}/{val_time}')

            # 每个epoch保存模型
            if epoch % 1 == 0:
                epoch_str = str(epoch).zfill(3)
                save_path = f"{train_params['Save_Path']}/model/model_{epoch_str}_{train_loss}_{val_loss}_.pth"
                torch.save(model.state_dict(), save_path)


    elif args.work == "Reconstruct_test":
        from src.TransVAE import *
        from tqdm import tqdm
        from nltk.translate.bleu_score import sentence_bleu
        import numpy as np


        def greedy_decode(model, mem, src_mask=None):
            start_symbol = w2i['<start>']
            max_len = params["tgt_len"]
            decoded = torch.ones(mem.shape[0], 1).fill_(start_symbol).long()
            tgt = torch.ones(mem.shape[0], max_len + 1).fill_(start_symbol).long()
            if src_mask != None:
                src_mask = src_mask.cuda()
            decoded = decoded.cuda()
            tgt = tgt.cuda()
            model.eval()
            for i in range(max_len):
                decode_mask = Variable(subsequent_mask(decoded.size(1)).long())
                decode_mask = decode_mask.cuda()
                out = model.decode(mem, src_mask, Variable(decoded), decode_mask)
                out = model.generator(out)
                prob = F.softmax(out[:, i, :], dim=-1)
                _, next_word = torch.max(prob, dim=1)
                next_word += 1
                tgt[:, i + 1] = next_word
                next_word = next_word.unsqueeze(1)
                decoded = torch.cat([decoded, next_word], dim=1)
            decoded = tgt[:, 1:]
            return decoded


        def decode_mols(encoded_tensors, org_dict):
            mols = []
            for i in range(encoded_tensors.shape[0]):
                encoded_tensor = encoded_tensors.cpu().numpy()[i, :] - 1
                mol_string = ''
                for i in range(encoded_tensor.shape[0]):
                    idx = encoded_tensor[i]
                    if org_dict[idx] == '<end>':
                        break
                    elif org_dict[idx] == '_':
                        pass
                    else:
                        mol_string += org_dict[idx]
                mols.append(mol_string)
            return mols


        def reconstruct(data, model, return_mems=False, return_str=True):
            with torch.no_grad():
                data = vae_data_gen(data, params["src_len"], char_dict=w2i)
                data_iter = torch.utils.data.DataLoader(data,
                                                        batch_size=train_params['BATCH_SIZE'],
                                                        shuffle=False, num_workers=0,
                                                        pin_memory=False, drop_last=False)
                batch_size = train_params['BATCH_SIZE']
                chunk_size = batch_size // train_params['BATCH_CHUNKS']
                model.eval()
                decoded_sequences = []
                decoded_properties = torch.empty((data.shape[0], 1))
                mems = torch.empty((data.shape[0], params['d_latent']))
                for j, data in enumerate(data_iter):
                    for i in range(train_params['BATCH_CHUNKS']):
                        batch_data = data[i * chunk_size:(i + 1) * chunk_size, :]
                        mols_data = batch_data[:, :-1]
                        src = Variable(mols_data).long()
                        src_mask = (src != w2i["_"]).unsqueeze(-2)
                        src = src.cuda()
                        src_mask = src_mask.cuda()
                        _, mem, _, _ = model.encode(src, src_mask)
                        props = torch.tensor(0)
                        start = j * batch_size + i * chunk_size
                        stop = j * batch_size + (i + 1) * chunk_size
                        decoded_properties[start:stop] = props
                        mems[start:stop, :] = mem.detach().cpu()
                        decoded = greedy_decode(mem=mem, model=model, src_mask=src_mask)
                        if return_str:
                            decoded = decode_mols(decoded, org_dict)
                            decoded_sequences += decoded
                        else:
                            decoded_sequences.append(decoded)
                if return_mems:
                    return decoded_sequences, decoded_properties, mems.detach().numpy()
                else:
                    return decoded_sequences, decoded_properties


        def calc_reconstruction_accuracies(input_sequences, output_sequences):
            "Calculates sequence, token and positional accuracies for a set of\
            input and reconstructed sequences"
            max_len = 126
            seq_accs = []
            hits = 0  # used by token acc only
            misses = 0  # used by token acc only
            position_accs = np.zeros((2, max_len))  # used by pos acc only
            for in_seq, out_seq in zip(input_sequences, output_sequences):
                if in_seq == out_seq:
                    seq_accs.append(1)
                else:
                    seq_accs.append(0)
                misses += abs(len(in_seq) - len(out_seq))  # number of missed tokens in the prediction seq
                for j, (token_in, token_out) in enumerate(
                        zip(in_seq, out_seq)):  # look at individual tokens for current seq
                    if token_in == token_out:
                        hits += 1
                        position_accs[0, j] += 1
                    else:
                        misses += 1
                    position_accs[1, j] += 1

            seq_acc = np.mean(seq_accs)  # list of 1's and 0's for correct or incorrect complete seq predictions
            token_acc = hits / (hits + misses)
            position_acc = []
            position_conf = []
            # calculating the confidence interval of the accuracy results
            z = 1.96  # 95% confidence interval
            for i in range(max_len):
                position_acc.append(position_accs[0, i] / position_accs[1, i])
                position_conf.append(z * math.sqrt(position_acc[i] * (1 - position_acc[i]) / position_accs[1, i]))

            seq_conf = z * math.sqrt(seq_acc * (1 - seq_acc) / len(seq_accs))
            # print(hits)
            # print(misses)
            token_conf = z * math.sqrt(token_acc * (1 - token_acc) / (hits + misses))

            return seq_acc, token_acc, position_acc, seq_conf, token_conf, position_conf


        data = pd.read_csv(args.vae_val_path).to_numpy()

        data_1D = data[:, 0]
        torch.backends.cudnn.benchmark = True

        model = create_VAE()
        model.load_state_dict(torch.load(args.vae_model_path))
        model.cuda()
        reconstructed_seq, props = reconstruct(data[:], model, return_mems=False)
        input_sequences = []
        for seq in data_1D:
            input_sequences.append(peptide_tokenizer(seq.upper()))
        output_sequences = []
        for seq in reconstructed_seq:
            output_sequences.append(peptide_tokenizer(seq.upper()))
        all_bleu = []
        for i in range(len(input_sequences)):
            tem_ref = data_1D[i]
            tem_can = reconstructed_seq[i]
            inp_ref = [[a for a in tem_ref]]
            pre_can = [b for b in tem_can]
            score = sentence_bleu(inp_ref, pre_can)
            all_bleu.append(score)
        bleu_score = np.array(all_bleu).mean()

        seq_accs, tok_accs, pos_accs, seq_conf, tok_conf, pos_conf = calc_reconstruction_accuracies(input_sequences,
                                                                                                    output_sequences)
        save_df = {}
        save_df['sequence accuracy'] = seq_accs
        save_df['sequence confidence'] = seq_conf
        save_df['token accuracy'] = tok_accs
        save_df['token confidence'] = tok_conf
        save_df['bleu_score'] = bleu_score
        print(save_df)
    # {'sequence accuracy': 0.9921936854972011, 'sequence confidence': 0.0004545871858095517, 'token accuracy': 0.9993429496509196, 'token confidence': 2.125677299529601e-05, 'bleu_score': 0.998939270668447}
    elif args.work == "GetMem_nc":
        from src.TransVAE import *
        from tqdm import tqdm

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # pd.read_csv(args.vae_train_path, nrows = 2000000)['Sequence'].to_numpy().reshape(-1, 1)
        data_train = pd.read_csv('./data/LatentDiffusion_Train.csv')['Sequence'].to_numpy().reshape(-1, 1)
        data_val = pd.read_csv('./data/LatentDiffusion_Val.csv')['Sequence'].to_numpy().reshape(-1, 1)
        train = vae_data_gen(data_train, params["src_len"], char_dict=w2i)
        val = vae_data_gen(data_val, params["src_len"], char_dict=w2i)

        train_properties = pd.read_csv('./data/LatentDiffusion_Train.csv')
        val_properties = pd.read_csv('./data/LatentDiffusion_Val.csv')
        # 选择第三列到最后一列的十个值（假设是第3到第12列，索引2到11）
        train_properties_values = train_properties.iloc[:, 2:12].to_numpy()
        val_properties_values = train_properties.iloc[:, 2:12].to_numpy()
        # print(train_mols_properties_values[0])
        # 创建一个 MinMaxScaler 对象
        # scaler = MinMaxScaler()
        #
        # # 对数据进行归一化
        # train_properties_values_normalized = scaler.fit_transform(train_properties_values)
        # val_properties_values_normalized = scaler.fit_transform(train_properties_values)
        train_dataset = MoleculeDataset(train, train_properties_values)
        val_dataset = MoleculeDataset(val, val_properties_values)
        model = create_VAE()
        model.load_state_dict(torch.load(args.vae_model_path))
        model.cuda()
        train_params['BATCH_SIZE'] = 2024
        print(len(train))
        print(len(val))


        def get_mem(data, model, save_path, type1):
            os.makedirs(f"{save_path}/{type1}", exist_ok=True)
            data_iter = torch.utils.data.DataLoader(
                data,
                batch_size=train_params['BATCH_SIZE'],
                shuffle=False, num_workers=0,
                pin_memory=False, drop_last=True
            )
            batch_size = train_params['BATCH_SIZE']
            chunk_size = batch_size // train_params['BATCH_CHUNKS']
            model.eval()
            for j, batch in tqdm(enumerate(data_iter)):
                data, property = batch
                property = property.clone().detach().to(torch.float32).to(device)
                for i in range(train_params["BATCH_CHUNKS"]):
                    batch_data = data[i * chunk_size: (i + 1) * chunk_size, :]
                    mols_data = batch_data[:, :-1]
                    mols_data = mols_data.cuda()
                    src = Variable(mols_data).long()
                    src_mask = (src != w2i["_"]).unsqueeze(-2)
                    pp_src = property.unsqueeze(1).expand(-1, src.size(1), -1).clone().detach().to(torch.float32).to(
                        device)
                    mem = model.encoder.get_mem(model.src_embed(src), src_mask, pp_src)
                    tem_num = len(os.listdir(f"{save_path}/{type1}"))
                    tem_mem = mem.detach().cpu().numpy()
                    np.save(f"{save_path}/{type1}/mem_{tem_num}.npy", tem_mem)


        get_mem(train_dataset, model, args.mem_save_path_nc, "train")
        get_mem(val_dataset, model, args.mem_save_path_nc, "val")
        print("run combine_mem.py before train")

    elif args.work == "GetMem_c":
        from src.TransVAE import *
        from tqdm import tqdm

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_pos = pd.read_csv('./data/posEcoli_guiyi.csv')['Sequence'].to_numpy().reshape(-1, 1)
        data_neg = pd.read_csv('./data/negEcoli_guiyi.csv')['Sequence'].to_numpy().reshape(-1, 1)
        # data_pos = pd.read_csv('./data/posEcoli_gai_guiyi.csv')['Sequence'].to_numpy().reshape(-1, 1)
        # data_neg = pd.read_csv('./data/neg_data_gai.csv')['Sequence'].to_numpy().reshape(-1, 1)

        pos = vae_data_gen(data_pos, params["src_len"], char_dict=w2i)
        neg = vae_data_gen(data_neg, params["src_len"], char_dict=w2i)
        pos_properties = pd.read_csv('./data/posEcoli_guiyi.csv')
        neg_properties = pd.read_csv('./data/negEcoli_guiyi.csv')
        # pos_properties = pd.read_csv('./data/posEcoli_gai_guiyi.csv')
        # neg_properties = pd.read_csv('./data/neg_data_gai.csv')
        # 选择第三列到最后一列的十个值（假设是第3到第12列，索引2到11）
        pos_properties_values = pos_properties.iloc[:, 2:12].to_numpy()
        neg_properties_values = neg_properties.iloc[:, 2:12].to_numpy()
        # print(train_mols_properties_values[0])
        # 创建一个 MinMaxScaler 对象
        # scaler = MinMaxScaler()
        #
        # # 对数据进行归一化
        # pos_properties_values_normalized = scaler.fit_transform(pos_properties_values)
        # neg_properties_values_normalized = scaler.fit_transform(neg_properties_values)
        pos_dataset = MoleculeDataset(pos, pos_properties_values)
        neg_dataset = MoleculeDataset(neg, neg_properties_values)

        model = create_VAE()
        model.load_state_dict(torch.load(args.vae_model_path))
        model.cuda()
        train_params['BATCH_SIZE'] = 1000  # 2024


        def get_mem(data, model, save_path, type1):
            os.makedirs(f"{save_path}/{type1}")
            data_iter = torch.utils.data.DataLoader(data,
                                                    batch_size=train_params['BATCH_SIZE'],
                                                    shuffle=False, num_workers=0,
                                                    pin_memory=False, drop_last=True)
            batch_size = train_params['BATCH_SIZE']
            chunk_size = batch_size // train_params['BATCH_CHUNKS']
            model.eval()
            for j, batch in tqdm(enumerate(data_iter)):
                data, property = batch
                property = property.clone().detach().to(torch.float32).to(device)
                for i in range(train_params["BATCH_CHUNKS"]):
                    batch_data = data[i * chunk_size: (i + 1) * chunk_size, :]
                    mols_data = batch_data[:, :-1]
                    props_data = batch_data[:, -1]
                    mols_data = mols_data.cuda()
                    props_data = props_data.cuda()
                    src = Variable(mols_data).long()
                    src_mask = (src != w2i["_"]).unsqueeze(-2)
                    pp_src = property.unsqueeze(1).expand(-1, src.size(1), -1).clone().detach().to(torch.float32).to(
                        device)
                    mem = model.encoder.get_mem(model.src_embed(src), src_mask, pp_src)
                    tem_num = len(os.listdir(f"{save_path}/{type1}"))
                    tem_mem = mem.detach().cpu().numpy()
                    np.save(f"{save_path}/{type1}/mem_{tem_num}.npy", tem_mem)


        os.makedirs(args.mem_save_path_c, exist_ok=True)
        get_mem(pos_dataset, model, args.mem_save_path_c, "pos_data_ecoli")
        get_mem(neg_dataset, model, args.mem_save_path_c, "neg_data_ecoli")
        print("run combine_mem_c.py before train")



    elif args.work == "LatentDiffusion_nocondition":

        from src.LateneDiffusion import *

        torch.backends.cudnn.benchmark = True

        init_process_group(backend='nccl')

        rank = dist.get_rank()

        device_id = rank % torch.cuda.device_count()

        num_steps = args.LatentDiffusion_num_steps

        batch_size = args.LatentDiffusion_batch_size

        shuffle = args.LatentDiffusion_shuffle

        num_workers = args.LatentDiffusion_num_workers

        pin_memory = args.LatentDiffusion_pin_memory

        drop_last = args.LatentDiffusion_drop_last

        learn_rate = args.LatentDiffusion_lr

        num_epochs = args.LatentDiffusion_epoch

        save_path = args.LatentDiffusion_save_path_nc

        model = DModel().to(device_id)

        model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

        diffusion = create_gaussian_diffusion(num_steps)

        sampler = UniformSampler(num_steps)

        train_data = MyDataset_nocond(f"./memory_single/train")

        val_data = MyDataset_nocond(f"./memory_single/val")

        train_sample = DistributedSampler(train_data)

        val_sample = DistributedSampler(val_data)

        train_loader = DataLoader(

            train_data,

            batch_size=batch_size,

            sampler=train_sample,

            shuffle=shuffle,

            num_workers=num_workers,

            pin_memory=pin_memory,

            drop_last=drop_last

        )

        val_loader = DataLoader(

            val_data,

            batch_size=int(batch_size / 10),

            sampler=val_sample,

            shuffle=shuffle,

            num_workers=num_workers,

            pin_memory=pin_memory,

            drop_last=drop_last

        )

        if rank == 0:

            os.makedirs(f"{save_path}", exist_ok=True)

            os.makedirs(f"{save_path}/model", exist_ok=True)

            log_filepath = f"{save_path}/train.log"

            try:

                f = open(log_filepath, 'r')

                f.close()

                already_wrote = True

            except FileNotFoundError:

                already_wrote = False

            log_file = open(log_filepath, 'a')

            if not already_wrote:
                log_file.write('epoch,batch_idx,data_type,tot_loss,run_time\n')

            log_file.close()

        opt = AdamW(model.parameters(), lr=learn_rate)

        if rank == 0:
            print(f"Train data : {len(train_data)}")

            print(f"Val data   : {len(val_data)}")

        best_loss = 1000

        for epoch in range(num_epochs):

            train_sample.set_epoch(epoch)

            train_loss = 0

            train_batch = 0

            start_time = time.time()

            model.train()

            start_batch_time = time.time()

            if rank == 0:

                train_run = tqdm(train_loader)

            else:

                train_run = train_loader

            start_time_abatch = time.time()

            for i, data in enumerate(train_run):

                b = data.shape[0]

                data = data.to(device_id)

                opt.zero_grad()

                time_steps, weights = sampler.sample(batch_size=data.shape[0], device_id=device_id)

                time_steps = time_steps.to(device_id)

                loss = diffusion.cal_loss(model, data, time_steps, rank=rank).mean()

                loss.backward()

                opt.step()

                train_loss += loss.item()

                if rank == 0:
                    batch_time = time.time() - start_batch_time

                    start_batch_time = time.time()

                    log_file = open(log_filepath, 'a')

                    log_file.write('{},{},{},{},{}\n'.format(

                        epoch,

                        i,

                        'train',

                        loss.item(),

                        batch_time

                    ))

                    log_file.close()

                train_batch += 1

            train_loss = train_loss / train_batch

            model.eval()

            val_loss = 0

            vak_batch = 0

            if rank == 0:

                val_run = tqdm(val_loader)

            else:

                val_run = val_loader

            for i, vdata in enumerate(val_run):

                seq = vdata

                seq = seq.to(device_id)

                time_steps, weights = sampler.sample(batch_size=seq.shape[0], device_id=device_id)

                time_steps = time_steps.to(device_id)

                loss = diffusion.cal_loss(model, seq, time_steps).mean()

                val_loss += loss.item()

                if rank == 0:
                    batch_time = time.time() - start_batch_time

                    start_batch_time = time.time()

                    log_file = open(log_filepath, 'a')

                    log_file.write('{},{},{},{},{}\n'.format(

                        epoch,

                        i,

                        'val',

                        loss.item(),

                        batch_time

                    ))

                    log_file.close()

                vak_batch += 1

            val_loss = val_loss / vak_batch

            if rank == 0:

                if val_loss < best_loss:
                    best_loss = val_loss

                    torch.save(model.module.state_dict(), f"{save_path}/best_model.pth")

                if epoch % 1 == 0:
                    torch.save(model.module.state_dict(),
                               f"{save_path}/model/model_{epoch}_{train_loss}_{val_loss}.pth")

                print(f"########################################################################################")

                print(f"Epoch {epoch}, Train loss : {round(train_loss, 5)}, Test loss : {round(val_loss, 5)}")

                print(f"Run time : {round(time.time() - start_time, 5)} s")


    elif args.work == "LatentDiffusion_condition":

        from src.LateneDiffusion import *

        torch.backends.cudnn.benchmark = True

        init_process_group(backend='nccl')

        rank = dist.get_rank()

        device_id = rank % torch.cuda.device_count()

        num_steps = args.LatentDiffusion_num_steps

        batch_size = args.LatentDiffusion_batch_size

        shuffle = args.LatentDiffusion_shuffle

        num_workers = args.LatentDiffusion_num_workers

        pin_memory = args.LatentDiffusion_pin_memory

        drop_last = args.LatentDiffusion_drop_last

        learn_rate = args.LatentDiffusion_lr

        num_epochs = args.LatentDiffusion_epoch_c

        save_path = args.LatentDiffusion_save_path_c

        model = DModel().to(device_id)

        model.load_state_dict(torch.load(f"{args.LatentDiffusion_save_path_nc}/best_model.pth"))

        model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

        diffusion = create_gaussian_diffusion(num_steps)

        sampler = UniformSampler(num_steps)

        all_data = MyDataset_cond(f"./memory_single_c/pos_data_ecoli", f"./memory_single_c/neg_data_ecoli")

        length = len(all_data)

        train_size, validate_size = int(0.8 * length), length - int(0.8 * length)

        train_data, val_data = random_split(all_data, [train_size, validate_size],
                                            generator=torch.Generator().manual_seed(42))

        train_sample = DistributedSampler(train_data)

        val_sample = DistributedSampler(val_data)

        train_loader = DataLoader(

            train_data,

            batch_size=batch_size,

            sampler=train_sample,

            shuffle=shuffle,

            num_workers=num_workers,

            pin_memory=pin_memory,

            drop_last=drop_last

        )

        val_loader = DataLoader(

            val_data,

            batch_size=int(batch_size / 10),

            sampler=val_sample,

            shuffle=shuffle,

            num_workers=num_workers,

            pin_memory=pin_memory,

            drop_last=drop_last

        )

        if rank == 0:

            os.makedirs(f"{save_path}", exist_ok=True)

            os.makedirs(f"{save_path}/model", exist_ok=True)

            log_filepath = f"{save_path}/train.log"

            try:

                f = open(log_filepath, 'r')

                f.close()

                already_wrote = True

            except FileNotFoundError:

                already_wrote = False

            log_file = open(log_filepath, 'a')

            if not already_wrote:
                log_file.write('epoch,batch_idx,data_type,tot_loss,run_time\n')

            log_file.close()

        opt = AdamW(model.parameters(), lr=learn_rate)

        if rank == 0:
            print(f"Train data : {len(train_data)}")

            print(f"Val data   : {len(val_data)}")

        best_loss = 1000

        for epoch in range(num_epochs):

            train_sample.set_epoch(epoch)

            train_loss = 0

            train_batch = 0

            start_time = time.time()

            model.train()

            start_batch_time = time.time()

            if rank == 0:

                train_run = tqdm(train_loader)

            else:

                train_run = train_loader

            start_time_abatch = time.time()

            for i, data__ in enumerate(train_run):

                data, label = data__

                b = data.shape[0]

                data = data.to(device_id)

                label = label.to(device_id)

                opt.zero_grad()

                time_steps, weights = sampler.sample(batch_size=data.shape[0], device_id=device_id)

                time_steps = time_steps.to(device_id)

                loss = diffusion.cal_loss(model, data, time_steps, label).mean()

                loss.backward()

                opt.step()

                train_loss += loss.item()

                if rank == 0:
                    batch_time = time.time() - start_batch_time

                    start_batch_time = time.time()

                    log_file = open(log_filepath, 'a')

                    log_file.write('{},{},{},{},{}\n'.format(

                        epoch,

                        i,

                        'train',

                        loss.item(),

                        batch_time

                    ))

                    log_file.close()

                train_batch += 1

            train_loss = train_loss / train_batch

            model.eval()

            val_loss = 0

            vak_batch = 0

            if rank == 0:

                val_run = tqdm(val_loader)

            else:

                val_run = val_loader

            for i, data__ in enumerate(val_run):

                vdata, label = data__

                seq = vdata

                seq = seq.to(device_id)

                label = label.to(device_id)

                time_steps, weights = sampler.sample(batch_size=seq.shape[0], device_id=device_id)

                time_steps = time_steps.to(device_id)

                loss = diffusion.cal_loss(model, seq, time_steps, label).mean()

                val_loss += loss.item()

                if rank == 0:
                    batch_time = time.time() - start_batch_time

                    start_batch_time = time.time()

                    log_file = open(log_filepath, 'a')

                    log_file.write('{},{},{},{},{}\n'.format(

                        epoch,

                        i,

                        'val',

                        loss.item(),

                        batch_time

                    ))

                    log_file.close()

                vak_batch += 1

            val_loss = val_loss / vak_batch

            if rank == 0:

                if val_loss < best_loss:
                    best_loss = val_loss

                    torch.save(model.module.state_dict(), f"{save_path}/best_model.pth")

                if epoch % 1 == 0:
                    torch.save(model.module.state_dict(),
                               f"{save_path}/model/model_{epoch}_{train_loss}_{val_loss}.pth")

                print(f"########################################################################################")

                print(f"Epoch {epoch}, Train loss : {round(train_loss, 5)}, Test loss : {round(val_loss, 5)}")

                print(f"Run time : {round(time.time() - start_time, 5)} s")


    elif args.work == "Generate":
        from src.TransVAE import *
        from src.LateneDiffusion import *


        class Generate():
            def __init__(self, steps, cond, model, batch_size, num_times):
                self.steps = steps
                self.num_steps = torch.tensor(steps).unsqueeze(0).cuda()
                self.class_name = torch.tensor(cond).unsqueeze(0).cuda()
                self.model = model
                self.diffusion = create_gaussian_diffusion(steps)
                self.batch_size = batch_size
                self.num_times = num_times

            def _scale_timesteps(self, t):
                return t.float() * (1000.0 / self.num_steps)

            def run_generate(self):
                model = self.model
                loop_func_ = self.diffusion.p_sample_loop_progressive_infill
                sample_shape = (self.batch_size, 127, 128)
                all_dat = []
                for _ in range(self.num_times):
                    run_out = self.generate(model, sample_shape, loop_func_)
                    all_dat.extend(run_out)
                return all_dat

            def generate(self, model, sample_shape, loop_func_):
                from tqdm import tqdm
                loop_func_ = tqdm(
                    loop_func_(
                        model,
                        sample_shape,
                        cond=self.class_name
                    ),
                )
                for sample in loop_func_:
                    final = sample["sample"]
                sample = final
                return sample


        def greedy_decode(model, mem, pp_src, src_mask=None):
            start_symbol = w2i['<start>']
            max_len = params["tgt_len"]
            decoded = torch.ones(mem.shape[0], 1).fill_(start_symbol).long()
            tgt = torch.ones(mem.shape[0], max_len + 1).fill_(start_symbol).long()
            if src_mask != None:
                src_mask = src_mask.cuda()
            decoded = decoded.cuda()
            tgt = tgt.cuda()
            model.eval()
            for i in range(max_len):
                decode_mask = Variable(subsequent_mask(decoded.size(1)).long())
                decode_mask = decode_mask.cuda()

                out = model.decode(mem, src_mask, Variable(decoded), decode_mask, pp_src)
                out = model.generator(out)
                prob = F.softmax(out[:, i, :], dim=-1)
                _, next_word = torch.max(prob, dim=1)
                next_word += 1
                tgt[:, i + 1] = next_word
                next_word = next_word.unsqueeze(1)
                decoded = torch.cat([decoded, next_word], dim=1)
            decoded = tgt[:, 1:]
            return decoded


        def sample(model, mem, src_mask, p, return_str=True):
            mem = mem.cuda()
            decoded = greedy_decode(model, mem, p, src_mask)
            if return_str:
                decoded = decode_mols(decoded, org_dict)
            return decoded


        Diffusion_model = DModel()
        Diffusion_model.load_state_dict(torch.load(args.Generate_Diffusion_model_path))
        Diffusion_model.cuda()
        Diffusion_model.eval()

        from tqdm import tqdm

        os.makedirs(args.Generate_tem_path)
        for _ in range(args.Generate_times):
            generator = Generate(args.LatentDiffusion_num_steps, args.Generate_condition, Diffusion_model,
                                 args.Generate_batch_num, args.Generate_batch_times)
            b = generator.run_generate()
            result = torch.empty([len(b), 127, 128])
            for i in range(len(b)):
                result[i, :] = b[i]
            num = len(os.listdir(f"{args.Generate_tem_path}"))
            torch.save(result, f"{args.Generate_tem_path}/{num}")
        file_list = os.listdir(f"{args.Generate_tem_path}")

        # exit()
        finally_result = []
        # 靶向某种细菌
        # properties = torch.tensor([[0.32100823, 0.73974641, 0.44248366, 0.23529412, 0.2097412, 0.13227273,
        #                             0.31768224, 0.35294118, 0.05882353, 0.]], dtype=torch.float32).cuda()
        # 平均值
        # properties = torch.tensor([[0.32100823, 0.73974641, 0.44248366, 0.23529412, 0.2097412, 0.13227273,
        #                             0.31768224, 0.35294118, 0.05882353, 0.]], dtype=torch.float32).cuda()
        properties = torch.tensor([[0.47759855, 0.81946014, 0.43799283, 0.23225806, 0.23186677, 0.08333333, 0.49376623,
                                    0.42580645, 0.29032258, 0.25806452]], dtype=torch.float32).cuda()
        properties = properties.unsqueeze(1)
        properties = properties.expand(128, 127, -1)

        for _ in tqdm(range(len(file_list))):
            result = torch.load(f"{args.Generate_tem_path}/{file_list[_]}")
            result = result / 25
            VAE_model = create_VAE()
            VAE_model.load_state_dict(torch.load(args.Generate_VAE_model_path))
            VAE_model.cuda()
            VAE_model.eval()
            output_dir = f"{args.Generate_mem_path}"
            os.makedirs(output_dir, exist_ok=True)  # exist_ok=True 确保路径已存在时不报错

            for i in range(int((args.Generate_batch_num * args.Generate_batch_times) / 128)):
                tem = result[i * 128: (i + 1) * 128, :]

                mem, _, _, pred_len = VAE_model.encoder.continue_encoder(tem.cuda())
                # mem_np = mem.cpu().detach().numpy()
                # print(mem_np.shape)
                # mem_file_path = f"{output_dir}/{i}.npy"
                # torch.save(mem_np, mem_file_path)
                # print(tem.shape) 128 127 128
                mask = (torch.arange(127)[None, :].cuda() < F.softmax(pred_len, dim=-1).argmax(dim=-1).unsqueeze(
                    -1)).unsqueeze(-2)
                # print(mem.shape,mask.shape) 128 128 ,128 1 127
                b = sample(VAE_model, mem, mask, properties, return_str=True)
                finally_result.extend(b)
        with open(f"{args.Generate_save_path}", "a") as wf:
            for idx, seq in enumerate(finally_result, start=1):  # 从1开始递增序号
                wf.write(f">seq{idx}\n")  # 写入FASTA头
                wf.write(f"{seq}\n")  # 写入序列内容


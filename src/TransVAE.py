import copy
import torch.nn.functional as F
import torch.nn as nn

import torch
import math
from torch.autograd import Variable
import numpy as np
import re
import pandas as pd
import os
import time












import pickle
with open('data/peptide_vocab.pkl', 'rb') as f:
    w2i = pickle.load(f)
char_weights = np.load('data/peptide_weight.npy')
org_dict = {}
for i, (k, v) in enumerate(w2i.items()):
    if i == 0:
        pass
    else:
        org_dict[int(v-1)] = k






params = {
    'h'                     :   4,
    'd_model'               :   128,
    'd_ff'                  :   512,
    'dropout'               :   0.1,
    'N'                     :   3,#3
    'd_latent'              :   128,
    'bypass_bottleneck'     :   False,
    'EPS_SCALE'             :   1,
    'd_model'               :   128,
    "src_len"               :   126,
    "tgt_len"               :   125,
    "vocab_size"            :   len(w2i.keys()),
}
def create_VAE(params = params):
    c = copy.deepcopy
    attn = MultiHeadedAttention(params['h'], params['d_model'])
    ff = PositionwiseFeedForward(params['d_model'], params['d_ff'], params['dropout'])
    position = PositionalEncoding(params['d_model'], params['dropout'])
    encoder = VAEEncoder(
        EncoderLayer(params['d_model'], params["src_len"], c(attn), c(ff), params['dropout']),
        params['N'], params['d_latent'], params['bypass_bottleneck'], params['EPS_SCALE'])
    decoder = VAEDecoder(
        EncoderLayer(params['d_model'], params["src_len"], c(attn), c(ff), params['dropout']),
        DecoderLayer(params['d_model'], params["tgt_len"], c(attn), c(attn), c(ff), params['dropout']),
        params['N'], params['d_latent'], params['bypass_bottleneck'], encoder.conv_bottleneck.conv_list)
    src_embed = nn.Sequential(Embeddings(params['d_model'], params["vocab_size"]), c(position))
    tgt_embed = nn.Sequential(Embeddings(params['d_model'], params["vocab_size"]), c(position))
    generator = Generator(params['d_model'], params["vocab_size"])
    property_predictor = None
    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator, property_predictor)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
def clones(module, N):
    """Produce N identical layers (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention' (adapted from Viswani et al.)"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
class EncoderDecoder(nn.Module):
    """
    Base transformer Encoder-Decoder architecture
    # 它能够接受源和目标序列，通过编码器编码源序列，通过解码器生成目标序列，并最终通过生成器输出最终的预测。
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, property_predictor):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.property_predictor = property_predictor

    def forward(self, src, tgt, pp_src, src_mask, tgt_mask):
        "Take in and process masked src and tgt sequences (added output shape from conv for decoder ease)"
        # src 512 127   tgt 512 126
        #print(type(pp_src))
        mem, mu, logvar, pred_len = self.encode(src, src_mask, pp_src)
        #print(mem.shape,mu.shape,logvar.shape,pred_len.shape) 都是  512 128    512-batch大小  128 d_model大小
        x , pred_pp = self.decode(mem, src_mask, tgt, tgt_mask, pp_src)
        #print(x.shape)  512 126 128
        x = self.generator(x)
        #print(x.shape)  512 126 22
        # if self.property_predictor is not None:
        #     prop = self.predict_property(mu, true_prop)
        # else:
        #     prop = None
        #print(self.encoder)
        #print(self.decoder)
        #print(self.src_embed)  包含一个Embedding层23---128和位置编码中引入dropout  防止过拟合
        #print(self.tgt_embed)  同上
        #print(self.generator) 简单的线性层  128-->22
        return x, mu, logvar, pred_len, pred_pp
        # , prop

    def encode(self, src, src_mask, pp_src):

        return self.encoder(self.src_embed(src), src_mask, pp_src)

    def decode(self, mem, src_mask, tgt, tgt_mask,  pp_src):
        return self.decoder(self.tgt_embed(tgt), mem, src_mask, tgt_mask,  pp_src)

    def predict_property(self, mu, true_prop):
        return self.property_predictor(mu, true_prop)
class VAEEncoder(nn.Module):
    "Base transformer encoder architecture"
    # N个transformer encoder部分 外加一个卷积瓶颈
    def __init__(self, layer, N, d_latent, bypass_bottleneck, eps_scale):
        super().__init__()
        self.layers = clones(layer, N)
        self.conv_bottleneck = ConvBottleneck(layer.size, layer.src_len)
        self.flat_conv_out = self.conv_bottleneck.conv_list[-1] * self.conv_bottleneck.out_channels
        self.z_means, self.z_var = nn.Linear(self.flat_conv_out, d_latent), nn.Linear(self.flat_conv_out, d_latent)
        self.norm = LayerNorm(layer.size)
        self.predict_len1 = nn.Linear(d_latent, d_latent*2)
        self.predict_len2 = nn.Linear(d_latent*2, layer.size)
        self.d_latent = d_latent
        self.bypass_bottleneck = bypass_bottleneck
        self.eps_scale = eps_scale
        # 新增多粒度门控组件
        # self.global_gate = nn.Sequential(  # 全局粒度
        #     nn.Linear(10, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 10),
        #     nn.Sigmoid()
        # )
        # self.local_gate = nn.Sequential(  # 时间步粒度
        #     nn.Linear(layer.size + 10, 32),  # 序列特征(128) + 属性(10)
        #     nn.ReLU(),
        #     nn.Linear(32, 10),
        #     nn.Sigmoid()
        # )
        # self.combine_gates = nn.Linear(10 * 2, 10)  # 融合双粒度信号
        self.pinjie = nn.Linear(d_latent+10,d_latent)

    def predict_mask_length(self, mem):
        "Predicts mask length from latent memory so mask can be re-created during inference"
        pred_len = self.predict_len1(mem)
        pred_len = self.predict_len2(pred_len)
        pred_len = F.softmax(pred_len, dim=-1)
        pred_len = torch.topk(pred_len, 1)[1]
        return pred_len

    def reparameterize(self, mu, logvar, eps_scale=1):
        "Stochastic reparameterization"
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) * eps_scale
        return mu + eps*std

    def forward(self, x, mask, pp_src):
        ### Attention and feedforward layers
        #print(x.shape) 512 127 128   127是src的length
        # 在这之前与属性进行拼接
        # batch_size, seq_len = x.size(0), x.size(1)
        # #print(pp_src.shape)
        # # 全局粒度门控（样本级）
        # global_gate = self.global_gate(pp_src)  # [batch,10]
        # #print(global_gate.shape)
        # global_gate = global_gate.unsqueeze(1)  # [batch, 1, 10]
        # #print(global_gate.shape)
        # global_gate = global_gate.expand(-1, x.size(1), -1)  # [batch, seq_len, 10]
        # #print(global_gate.shape)
        # # 局部粒度门控（时间步级）
        seq_len = x.size(1)
        pp_expanded = pp_src.unsqueeze(1).expand(-1, seq_len, -1)  # [batch,seq,10]
        # #print(pp_expanded.shape)
        # local_input = torch.cat([x, pp_expanded], dim=-1)  # [batch,seq,138]
        # local_gate = self.local_gate(local_input)  # [batch,seq,10]
        #
        # # 融合双粒度信号
        # combined_gate = torch.cat([global_gate, local_gate], dim=-1)  # [batch,seq,20]
        # final_gate = self.combine_gates(combined_gate)  # [batch,seq,10]
        #
        # # 应用加权
        # weighted_pp = pp_expanded * final_gate  # [batch,seq,10]
        #
        # # 拼接并降维
        # x = torch.cat((x, weighted_pp), dim=-1)  # [batch,seq,138]
        # x = self.pinjie(x)  # [batch,seq,d_latent]
        #pp_src = torch.tensor(pp_src)
        x = torch.cat((x, pp_expanded), dim=-1)
        #mask = torch.cat((mask,property_mask),dim=-1)
        x = self.pinjie(x)
        for i, attn_layer in enumerate(self.layers):
            x = attn_layer(x, mask)
        ### Batch normalization
        #print(x.shape) 512 127 128
        mem = self.norm(x)
        #print(mem.shape) 512 127 128
        ### Convolutional Bottleneck
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
        else:
            mem = mem.permute(0, 2, 1)
            #print(mem.shape) 512 128 127
            mem = self.conv_bottleneck(mem)
            #print(mem.shape) 512 64 9
            mem = mem.contiguous().view(mem.size(0), -1)
            #print(mem.shape) 512 576
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar, self.eps_scale)
            pred_len = self.predict_len1(mu)
            pred_len = self.predict_len2(pred_len)
        return mem, mu, logvar, pred_len
    def get_mem(self, x, mask, pp_src):
        # x = torch.cat((x, pp_src), dim=-1)
        # # mask = torch.cat((mask,property_mask),dim=-1)
        # x = self.pinjie(x)
        # batch_size, seq_len = x.size(0), x.size(1)
        #
        # # 全局粒度门控（样本级）
        # global_gate = self.global_gate(pp_src)  # [batch,10]
        # global_gate = global_gate.unsqueeze(1)  # [batch, 1, 10]
        # # print(global_gate.shape)
        # global_gate = global_gate.expand(-1, x.size(1), -1)  # [batch, seq_len, 10]
        #
        # # 局部粒度门控（时间步级）
        seq_len = x.size(1)
        #print(pp_src.shape)
        #pp_expanded = pp_src.unsqueeze(1).expand(-1, seq_len, -1)  # [batch,seq,10]
        # local_input = torch.cat([x, pp_expanded], dim=-1)  # [batch,seq,138]
        # local_gate = self.local_gate(local_input)  # [batch,seq,10]
        #
        # # 融合双粒度信号
        # combined_gate = torch.cat([global_gate, local_gate], dim=-1)  # [batch,seq,20]
        # final_gate = self.combine_gates(combined_gate)  # [batch,seq,10]
        #
        # # 应用加权
        # weighted_pp = pp_expanded * final_gate  # [batch,seq,10]

        # 拼接并降维
        # x = torch.cat((x, weighted_pp), dim=-1)  # [batch,seq,138]
        # x = self.pinjie(x)  # [batch,seq,d_latent]

        x = torch.cat((x, pp_src), dim=-1)
        # mask = torch.cat((mask,property_mask),dim=-1)
        x = self.pinjie(x)
        for i, attn_layer in enumerate(self.layers):
            x = attn_layer(x, mask)
        mem = self.norm(x)
        return mem
    def continue_encoder(self,mem):
        mem = mem.permute(0, 2, 1)
        mem = self.conv_bottleneck(mem)
        mem = mem.contiguous().view(mem.size(0), -1)
        mu, logvar = self.z_means(mem), self.z_var(mem)
        mem = self.reparameterize(mu, logvar, self.eps_scale)
        pred_len = self.predict_len1(mu)
        pred_len = self.predict_len2(pred_len)
        return mem, mu, logvar, pred_len
    def forward_w_attn(self, x, mask, pp_src):
        "Forward pass that saves attention weights"
        # x = torch.cat((x, pp_src), dim=-1)
        # # mask = torch.cat((mask,property_mask),dim=-1)
        # x = self.pinjie(x)
        # batch_size, seq_len = x.size(0), x.size(1)
        #
        # # 全局粒度门控（样本级）
        # global_gate = self.global_gate(pp_src)  # [batch,10]
        # global_gate = global_gate.unsqueeze(1)  # [batch, 1, 10]
        # # print(global_gate.shape)
        # global_gate = global_gate.expand(-1, x.size(1), -1)  # [batch, seq_len, 10]
        #
        # # 局部粒度门控（时间步级）
        # pp_expanded = pp_src.unsqueeze(1).expand(-1, seq_len, -1)  # [batch,seq,10]
        # local_input = torch.cat([x, pp_expanded], dim=-1)  # [batch,seq,138]
        # local_gate = self.local_gate(local_input)  # [batch,seq,10]
        #
        # # 融合双粒度信号
        # combined_gate = torch.cat([global_gate, local_gate], dim=-1)  # [batch,seq,20]
        # final_gate = self.combine_gates(combined_gate)  # [batch,seq,10]
        #
        # # 应用加权
        # weighted_pp = pp_expanded * final_gate  # [batch,seq,10]
        #
        # # 拼接并降维
        # x = torch.cat((x, weighted_pp), dim=-1)  # [batch,seq,138]
        # x = self.pinjie(x)  # [batch,seq,d_latent]
        x = torch.cat((x, pp_src), dim=-1)
        # mask = torch.cat((mask,property_mask),dim=-1)
        x = self.pinjie(x)
        attn_wts = []
        for i, attn_layer in enumerate(self.layers):
            x, wts = attn_layer(x, mask, return_attn=True)
            attn_wts.append(wts.detach().cpu())
        mem = self.norm(x)
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
        else:
            mem = mem.permute(0, 2, 1)
            mem = self.conv_bottleneck(mem)
            mem = mem.contiguous().view(mem.size(0), -1)
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar, self.eps_scale)
            pred_len = self.predict_len1(mu)
            pred_len = self.predict_len2(pred_len)
        return mem, mu, logvar, pred_len, attn_wts
class EncoderLayer(nn.Module):
    "Self-attention/feedforward implementation"
    def __init__(self, size, src_len, self_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.src_len = src_len
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 2)

    def forward(self, x, mask, return_attn=False):
        if return_attn:
            attn = self.self_attn(x, x, x, mask, return_attn=True)
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward), attn
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward)
class VAEDecoder(nn.Module):
    "Base transformer decoder architecture"
    def __init__(self, encoder_layers, decoder_layers, N, d_latent, bypass_bottleneck, conv_list):
        super().__init__()
        self.final_encodes = clones(encoder_layers, 1)
        self.layers = clones(decoder_layers, N)
        self.norm = LayerNorm(decoder_layers.size)
        self.bypass_bottleneck = bypass_bottleneck
        self.size = decoder_layers.size
        self.tgt_len = decoder_layers.tgt_len
        self.conv_out = conv_list[-1] #take the last outputs shape from the convlution
        # Reshaping memory with deconvolution
        self.deconv_bottleneck = DeconvBottleneck(decoder_layers.size, encoder_layers.src_len, conv_list) #反卷积瓶颈，用于对潜在空间表示进行上采样（通过反卷积）。其目的是将潜在空间的维度映射到更高的维度。
        self.linear = nn.Linear(d_latent, 64*self.conv_out)
        self.pinjie = nn.Linear(d_latent + 10, d_latent)
        self.decoder_fc = nn.Sequential(
            nn.Linear(d_latent*126, d_latent),
            nn.PReLU(),
            #nn.Dropout(p=0.2),   #第二版修改位置 添加dropout
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, 10)
        )

    def forward(self, x, mem, src_mask, tgt_mask, pp_src):
        #(self.tgt_embed(tgt), mem, src_mask, tgt_mask, pp_src)
        ### Deconvolutional bottleneck (up-sampling)
        if not self.bypass_bottleneck:
            mem = F.relu(self.linear(mem))
            mem = mem.view(-1, 64, self.conv_out)
            mem = self.deconv_bottleneck(mem)
            mem = mem.permute(0, 2, 1)
        ### Final self-attention layer
        for final_encode in self.final_encodes:
            mem = final_encode(mem, src_mask)
        # Batch normalization
        mem = self.norm(mem)
        #print(mem.shape,pp_src.shape)#torch.Size([128, 127, 128]) torch.Size([128, 10])
        seq_len = mem.size(1)
        # 生成时不用改维度
        #pp_expanded = pp_src.unsqueeze(1).expand(-1, seq_len, -1)
        pp_expanded = pp_src  #生成时改回来 把上面注释掉
        mem = torch.cat((mem, pp_expanded), dim=-1)
        #mem = torch.cat((mem, pp_src), dim=-1)
        mem = self.pinjie(mem)
        ### Source-attention layers
        # x = torch.cat((x, pp_tgt), dim=-1)
        # # mask = torch.cat((mask,property_mask),dim=-1)
        # x = self.pinjie(x)
        for i, attn_layer in enumerate(self.layers):
            x = attn_layer(x, mem, mem, src_mask, tgt_mask)
        xx = x.view(x.size(0), -1)
        # 生成时注释掉pred_pp
        #pred_pp = self.decoder_fc(xx)
        return self.norm(x)#, pred_pp

    def forward_w_attn(self, x, mem, src_mask, tgt_mask, pp_src):
        "Forward pass that saves attention weights"

        if not self.bypass_bottleneck:
            mem = F.relu(self.linear(mem))
            mem = mem.view(-1, 64, self.conv_out)
            mem = self.deconv_bottleneck(mem)
            mem = mem.permute(0, 2, 1)
        for final_encode in self.final_encodes:
            mem, deconv_wts  = final_encode(mem, src_mask, return_attn=True)
        mem = self.norm(mem)
        seq_len = mem.size(1)
        pp_expanded = pp_src.unsqueeze(1).expand(-1, seq_len, -1)
        mem = torch.cat((mem, pp_expanded), dim=-1)
        mem = self.pinjie(mem)
        # x = torch.cat((x, pp_tgt), dim=-1)
        # # mask = torch.cat((mask,property_mask),dim=-1)
        # x = self.pinjie(x)
        src_attn_wts = []
        for i, attn_layer in enumerate(self.layers):
            x, wts = attn_layer(x, mem, mem, src_mask, tgt_mask, return_attn=True)
            src_attn_wts.append(wts.detach().cpu())
        xx = x.view(x.size(0), -1)
        pred_pp = self.decoder_fc(xx)
        return self.norm(x), [deconv_wts.detach().cpu()], src_attn_wts, pred_pp
class DecoderLayer(nn.Module):
    "Self-attention/source-attention/feedforward implementation"
    def __init__(self, size, tgt_len, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.tgt_len = tgt_len
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 3)

    def forward(self, x, memory_key, memory_val, src_mask, tgt_mask, return_attn=False):
        m_key = memory_key
        m_val = memory_val
        if return_attn:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            src_attn = self.src_attn(x, m_key, m_val, src_mask, return_attn=True)
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m_key, m_val, src_mask))
            return self.sublayer[2](x, self.feed_forward), src_attn
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m_key, m_val, src_mask))
            return self.sublayer[2](x, self.feed_forward)
class MultiHeadedAttention(nn.Module):
    "Multihead attention implementation (based on Vaswani et al.)"
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads"
        super().__init__()
        assert d_model % h == 0
        #We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, return_attn=False):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if return_attn:
            return self.attn
        else:
            return self.linears[-1](x)
class PositionwiseFeedForward(nn.Module):
    "Feedforward implementation"
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
class PositionalEncoding(nn.Module):
    "Static sinusoidal positional encoding layer"
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
class ConvBottleneck(nn.Module):
    """
    Set of convolutional layers to reduce memory matrix to single
    latent vector
    NEED TO MAKE THIS GENERALIZEABLE IT IS HARD SET TO 64*9 = 576 from an input vector of length 128 
    """
    def __init__(self, size, src_len):
        super().__init__()
        conv_layers = []
        self.conv_list = [] # this will allow a flexible model input by changing the decoder shape to match each level of convolution
        in_d = size
        first = True
        input_shape = src_len
        self.out_channels = 64
        for i in range(3):
            out_d = int((in_d - 64) // 2 + 64)
            if first:
                kernel_size = 9 #OG_9
                first = False
            else:
                kernel_size = 8 #OG_8
            if i == 2:
                out_d = self.out_channels
            conv_layers.append(nn.Sequential(nn.Conv1d(in_d, out_d, kernel_size), nn.MaxPool1d(kernel_size=2)))
            in_d = out_d
            #conv_out_shape [(W−K+2P)/S]+1 ;W:input, K:kernel_size, P:padding, S:stride default=1
            #maxpool output shape [(W+2p-D*(K-1)-1)/S]+1  W:input, D:dilation, K:kernel_size, P:padding, S:stride default=kernel_size
            conv_out_shape = ((input_shape-kernel_size)//1)+1 
            maxpool_out_shape = ((conv_out_shape-(2-1)-1)//2)+1
            input_shape = maxpool_out_shape
            self.conv_list.append(input_shape)#save the output shape
        self.conv_layers = ListModule(*conv_layers)

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        return x
class DeconvBottleneck(nn.Module):
    """
    Set of deconvolutional layers to reshape latent vector
    back into memory matrix
    """
    def __init__(self, size, src_len, conv_list):
        super().__init__()
        deconv_layers = []
        in_d = 64
        input_shape = src_len+1
        conv_list.insert(0,input_shape) #add the original source length to the conv shape list
        for i in range(3):
            #formula to find appropriate kernel size for each layer:(L_out-1)-2(L_in-1)+1=K ,K:kernel_size,L_out:new_shape,L_in:old_shape
            L_in = conv_list[3-i]
            L_out = conv_list[3-(i+1)]
            out_d = (size - in_d) // 4 + in_d
            stride = 2
            kernel_size = (L_out-1)-2*(L_in-1)+1
            if i == 2:
                out_d = size
            deconv_layers.append(nn.Sequential(nn.ConvTranspose1d(in_d, out_d, kernel_size,
                                                                  stride=stride, padding=0)))
            in_d = out_d
        self.deconv_layers = ListModule(*deconv_layers)

    def forward(self, x):
        for deconv in self.deconv_layers:
            x = F.relu(deconv(x))
        return x
class Generator(nn.Module):
    "Generates token predictions after final decoder layer"
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab-1)

    def forward(self, x):
        return self.proj(x)
class Embeddings(nn.Module):
    "Transforms input token id tensors to size d_model embeddings. Importantly this embedding is learnable! Weights change with backprop."
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model) #Square root is for the transformer model to keep num's low
class LayerNorm(nn.Module):
    "Construct a layernorm module (manual)"
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))
class ListModule(nn.Module):
    """Create single pytorch module from list of modules"""
    def __init__(self, *args):
        super().__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)
class KLAnnealer:
    """
    Scales KL weight (beta) linearly according to the number of epochs
    """
    def __init__(self, kl_low, kl_high, n_epochs, start_epoch):
        self.kl_low = kl_low
        self.kl_high = kl_high
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch

        self.kl = (self.kl_high - self.kl_low) / (self.n_epochs - self.start_epoch)

    def __call__(self, epoch):
        if self.start_epoch == 0:
            k = (epoch - self.start_epoch) if epoch >= self.start_epoch else 0
            beta = self.kl_low + k * self.kl
            if beta > self.kl_high:
                beta = self.kl_high  
            else:
                pass
        else: #when checkpointing just set the beta to the max value from previous training
            beta = self.kl_high
        return beta
class NoamOpt:
    "Optimizer wrapper that implements rate decay (adapted from\
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"
    def __init__(self, model_size, factor, warmup, optimizer,lr =0.001):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

        self.state_dict = self.optimizer.state_dict()
        self.state_dict['step'] = 0
        self.state_dict['rate'] = 0
        self.lr = lr

    def step(self):
        "Update parameters and rate"
        self.state_dict['step'] += 1
        if self.lr == None:
            rate = self.rate()
        else:
            rate = self.lr

        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.state_dict['rate'] = rate
        self.optimizer.step()
        for k, v in self.optimizer.state_dict().items():
            self.state_dict[k] = v

    def rate(self, step=None):
        "Implement 'lrate' above"
        if step is None:
            step = self.state_dict['step']
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
# w2i = {
#          '<start>': 0,
#          '<end>': 1,
#          '_': 2,        # padding
#          'm': 3,        # 蛋氨酸
#          'a': 4,        # 丙氨酸
#          'f': 5,        # 苯丙氨酸
#          's': 6,        # 丝氨酸
#          'e': 7,        # 谷氨酸
#          'd': 8,        # 天冬氨酸
#          'v': 9,        # 缬氨酸
#          'l': 10,       # 亮氨酸
#          'k': 11,       # 赖氨酸
#          'y': 12,       # 酪氨酸
#          'r': 13,       # 精氨酸
#          'p': 14,       # 脯氨酸
#          'n': 15,       # 天冬酰胺
#          'w': 16,       # 色氨酸
#          'q': 17,       # 谷氨酰胺
#          'c': 18,       # 半胱氨酸
#          'g': 19,       # 甘氨酸
#          'i': 20,       # 异亮氨酸
#          'h': 21,       # 组氨酸
#          't': 22,       # 苏氨酸

# }
# i2w = {v:k for k,v in w2i.items()}

def peptide_tokenizer(peptide):
    "Tokenizes SMILES string"
    # 检查peptide类型
    # print(peptide)
    # print(type(peptide))
    # if not isinstance(peptide, str):
    #     peptide = str(peptide)  # 将非字符串类型转换为字符串
    #     print(f"Warning: Converted {peptide} to string")
    #need to remove "X", "B", "Z", "U", "O"
    pattern =  "(G|A|L|M|F|W|K|Q|E|S|P|V|I|C|Y|H|R|N|D|T)"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(peptide)]
    assert peptide == ''.join(tokens), ("{} could not be joined".format(peptide))
    return tokens
def encode_seq(sequence, max_len, char_dict):
    "Converts tokenized sequences to list of token ids"
    for i in range(max_len - len(sequence)):
        if i == 0:
            sequence.append('<end>')
        else:
            sequence.append('_')
    seq_vec = [char_dict[c] for c in sequence]
    return seq_vec
def vae_data_gen(data, max_len=126, char_dict=w2i):
    seq_list = data[:,0]

    props = np.zeros(seq_list.shape)
    del data
    seq_list = [peptide_tokenizer(x) for x in seq_list]
    
    encoded_data = torch.empty((len(seq_list), max_len+2))
    for j, seq in enumerate(seq_list):
        encoded_seq = encode_seq(seq, max_len, char_dict)
        encoded_seq = [0] + encoded_seq
        encoded_data[j,:-1] = torch.tensor(encoded_seq)
        encoded_data[j,-1] = torch.tensor(props[j])
    return encoded_data
def subsequent_mask(size):
    """Mask out subsequent positions (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
def make_std_mask(tgt, pad):
    """
    Creates sequential mask matrix for target input (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)

    Arguments:
        tgt (torch.tensor, req): Target vector of token ids
        pad (int, req): Padding token id
    Returns:
        tgt_mask (torch.tensor): Sequential target mask
    """
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask
def trans_vae_loss(x, x_out, mu, logvar, true_len, pred_len, weights, property, pred_pp, beta = 1):
    "Binary Cross Entropy Loss + Kullbach leibler Divergence + Mask Length Prediction"

    x = x.long()[:,1:] - 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    true_len = true_len.contiguous().view(-1)
    # print(x.shape)
    # print(x_out.shape)
    BCEmol = F.cross_entropy(x_out, x, reduction='mean', weight=weights)
    BCEmask = F.cross_entropy(pred_len, true_len, reduction='mean')
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    bce_prop = torch.tensor(0.)
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    # 定义均方误差损失函数
    mse_loss = nn.MSELoss()
    # 计算损失
    PP_loss = mse_loss(property, pred_pp)
    return BCEmol + BCEmask + KLD + bce_prop + PP_loss, BCEmol, BCEmask, KLD, PP_loss
    # , bce_prop

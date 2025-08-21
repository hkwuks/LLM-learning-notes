import math
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, d_model):
        super(Embedding, self).__init__()

        self.ebd = nn.Embedding(num_embeddings, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.ebd(x) * math.sqrt(self.d_model)


class PositionWiseFFN(nn.Module):
    '''
    基于位置的前馈网络
    '''

    def __init__(self, input, hiddens, outputs, **kwargs):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(input, hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hiddens, outputs)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))


class AddNorm(nn.Module):
    '''
    残差连接后进行层规范化
    '''

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EncoderBlock(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,norm_shape,input,hiddens,num_heads,dropout,use_bias=False):
        super(EncoderBlock, self).__init__()
        self.attention=

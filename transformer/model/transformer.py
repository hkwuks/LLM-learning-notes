import torch
from torch import nn

from transformer.model.decoder import Decoder
from transformer.model.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * self.make_no_peak_mask(trg, trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
        return output

    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        len_q, len_k = q.size(1), k.size(1)

        # [batch_size,1,1,len_k]
        k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)  # torch.ne: not equal
        # [batch_size,1,len_q,len_k]
        k = k.repeat(1, 1, 1, len_k)

        # [batch_size,1,len_q,1]
        q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
        # [batch_size,1,len_q,len_k]
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        # [len_q,len_k]
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolType).to(self.device)  # torch.tril: 返回矩阵的下三角
        return mask

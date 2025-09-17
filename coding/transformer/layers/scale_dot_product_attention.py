import math
from torch import nn


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None, e=1e-12):
        '''
        :param Q: [batch_size, n_heads, seq_length, dim_k]
        :param K:
        :param V:
        :param mask:
        :param e:
        :return:
        '''
        batch, head, length, dim = K.size()

        K_t = K.transpose(2, 3)
        score = (Q @ K_t) / math.sqrt(dim)  # @表示矩阵相乘，相当于torch.bmm()

        if mask is not None:
            score.masked_fill_(mask == 0, -float('inf'))

        score = self.softmax(score)

        V = score @ V
        return V, score

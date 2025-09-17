from torch import nn

from transformer.embedding.positional_encoding import PositionalEncoding
from transformer.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    '''
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    '''

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        '''
        :param vocab_size:
        :param d_model:
        :param max_len:
        :param drop_prob:
        :param device:
        '''
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)

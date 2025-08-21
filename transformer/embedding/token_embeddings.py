from torch import nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        '''
        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        '''
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

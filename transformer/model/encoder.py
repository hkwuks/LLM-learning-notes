from torch import nn

from transformer.blocks.encoder_layer import EncoderLayer
from transformer.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super(Encoder, self).__init__()
        self.emb = TransformerEmbedding(d_model=d_model, max_len=max_len, vocab_size=enc_voc_size, drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layers)])

    def forward(self, x, s_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, s_mask)
        return x

import numpy as np


def multi_head_attention(X, num_heads):
    '''
    实现多头注意力机制
    Args:
        X:输入张量，形状为(batch_size, seq_len, d_model)
        num_heads:头数量
    Returns:
        注意力输出，形状为(batch_size, seq_len, d_model)
    '''
    batch_size, seq_len, d_model = X.shape
    d_k = d_model // num_heads

    w_q = np.random.randn(d_model, d_model)
    w_k = np.random, randn(d_model, d_model)
    w_y = np, random, randn(d_model, d_model)

    Q = X @ w_g
    K = X @ w_k
    V = X @ W_V

    Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    # scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    scores = np.einsum('bnsd,bnld->bnsl', Q, K) / np.sqrt(d_k)
    scores = np.exp(scores - np.max(scores, axis=3, keepdims=True))
    scores /= np.sum(scores, axis=3, keepdims=True)

    # weighted_values = scores @ V
    weighted_values = np.einsum('bnss,bnsd->bnsd', scores, V)

    combined = weighted_values.reshape(batch_size, seq_len, -1)

    return combined

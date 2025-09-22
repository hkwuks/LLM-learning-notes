import numpy as np


def MultiQueryAttention(X, mask=None, n_heads=8):
    '''
        实现多查询共享注意力机制
        Args:
            X:输入张量，形状为(batch_size, seq_len, d_model)
            mask: 掩码张量，形状为(batch_size, seq_len, seq_len)，mask的作用是掩盖padding或掩盖后面的Token
            num_heads: 头数量
        Returns:
            注意力输出，形状为(batch_size, seq_len, d_model)
        '''
    batch_size, seq_len, d_model = X.shape
    d_k = d_model // n_heads

    w_q = np.random.randn(d_model, d_model)
    w_k = np.random.randn(d_model, d_k)
    w_v = np.random.randn(d_model, d_k)

    Q = X @ w_q
    K = X @ w_k
    V = X @ w_v

    Q = Q.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, 1, d_k).repeat(n_heads, axis=2).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, 1, d_k).repeat(n_heads, axis=2).transpose(0, 2, 1, 3)

    # scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    scores = np.einsum('bnsd,bnld->bnsl', Q, K) / np.sqrt(d_k)
    scores = np.exp(scores - np.max(scores, axis=3, keepdims=True))
    scores /= np.sum(scores, axis=3, keepdims=True)

    if mask is not None:
        scores = np.where(mask == 0, scores, -np.inf)

    # weighted_values = scores @ V
    weighted_values = np.einsum('bnss,bnsd->bnsd', scores, V)

    combined = weighted_values.reshape(batch_size, seq_len, -1)

    return combined

matrix = np.random.randn(10, 10, 64)
output = MultiQueryAttention(matrix)
print(output)
print(output.shape)
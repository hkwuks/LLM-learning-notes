import numpy as np
from scipy.sparse import csr_matrix


def SparseMatrixMultiply(A: np.ndarray, B: np.ndarray):
    '''
    稀疏矩阵乘法
    Args:
        A: 稀疏矩阵
        B: 被乘矩阵

    Returns:

    '''
    a_shape = A.shape
    b_shape = B.shape

    if len(a_shape) != len(b_shape):
        raise ValueError('A and B must have same shape')
    if len(a_shape) < 2 or len(b_shape) < 2:
        raise ValueError('The shape of A and B must be at least 2')

    A = A.reshape(-1, a_shape[-2], a_shape[-1])
    B = B.reshape(-1, a_shape[-2], a_shape[-1])  # 因先进行数值乘法然后再统计累加，因此不做转置

    if A.shape[0] != B.shape[0]:
        raise ValueError('A and B must have same number of 2-D arrays')

    results = []
    if len(A.shape) > 2:
        for i in range(A.shape[0]):
            results.append(SparseMatrixMultiply2D(A[i], B[i]))
    else:
        results = SparseMatrixMultiply2D(A, B)

    # todo: 还原矩阵形状（其实csr格式是可能丢失矩阵形状信息的）
    return np.asarray(results)


def SparseMatrixMultiply2D(A: np.ndarray, B: np.ndarray):
    a_data, a_indices, a_indptr = to_csr(A)
    # 生成行索引数据
    row_indices = np.repeat(np.arange(len(a_indptr) - 1), np.diff(a_indptr))

    # 计算每个非零元素的贡献值
    all_indices = [j * max(a_indices + 1) + col for j, s, l in
                   zip(range(len(a_indptr) - 1), a_indptr, np.diff(a_indptr))
                   for col in a_indices[s:s + l]]
    contributions = a_data * B.reshape(-1)[all_indices]
    return np.bincount(row_indices, weights=contributions)


def to_csr(A: np.ndarray):
    '''
    将二维矩阵转换为CSR格式
    Args:
        A: 二维矩阵

    Returns:

    '''
    if len(A.shape) != 2:
        raise ValueError('The shape of A must be 2-D')

    data = []
    indices = []
    indptr = [0]  # 每行的起始索引

    for row in A:
        non_zero_vals = row[row != 0]
        non_zero_indices = np.where(row != 0)[0]

        data.extend(non_zero_vals.tolist())
        indices.extend(non_zero_indices.tolist())
        indptr.append(indptr[-1] + len(non_zero_vals))
    return np.array(data), np.array(indices), np.array(indptr)


X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
X = X[np.newaxis, :]
print(X)
print(X.shape)

result = SparseMatrixMultiply(X, X)
print(result)
print(result.shape)

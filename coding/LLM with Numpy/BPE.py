import re,collections

def get_stats(vocab):
    '''
    统计词元对频率
    Args:
        vocab:

    Returns:

    '''
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split() # 这里不是真实的分割操作，只是一个示例
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    '''
    合并词元对
    Args:
        pair:
        v_in:

    Returns:

    '''
    v_out={}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        
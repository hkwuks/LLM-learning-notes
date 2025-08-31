# 混合检索

混合检索（Hybrid Search）是一种结合了**稀疏向量（Sparse Vectors）** 和 **稠密向量（Dense Vectors）** 优势的先进的搜索技术。旨在同时利用稀疏向量的关键词精确匹配能力和稠密向量的语义理解能力，以克服单一向量检索的局限性，从而在各种搜索场景下提供更准确、更鲁棒的搜索结果。 

## 稀疏向量 vs 稠密向量

**稀疏向量**也被称为“词法向量”，是基于词频统计的传统信息检索方法的数学表示。它通常是一个维度极高（与词汇表大小相当）但绝大多数元素为零的向量。它采用精准的“词袋”匹配模型，将文档式微一堆词的集合，不考虑顺序和语法。其中向量的每一个维度都直接对应一个具体的词，非零值则代表该词在文档中的重要性（权重）。这类向量的典型代表是TF-IDF和BM25。其中BM25是目前最成功、应用最广泛的稀疏向量计分算法之一，其核心公式如下：

$$Score(Q,D)=\sum_{i=1}^{n}IDF(q_{i})\cdot\frac{f(q_{i},D)\cdot(k_{1}+1)}{f(q_{i},D)+k_{1}\cdot(1-b+b\cdot\frac{|D|}{avgdl})}$$

其中：

- $IDF(q_i)$：查询词$q_i$的逆文档频率，用于衡量一个词的普遍程度。越常见的词，IDF值越低。
- $f(q_i, D)$：查询词$q_i$在文档D中的词频。
- $|D|$：文档$D$的长度。
- $avgdl$：集合中所有文档的平均长度。
- $k_1, b$：可调节的超参数。$k_i$用于控制词频饱和度（一个词在文档中出现10次和100次，其重要性增长并非线性），$b$用于控制文档长度归一化程度。

这种方法的优点是可解释性强（每个维度都代表一个确切的词），无需训练，能够实现关键词的精确匹配，对于专业术语和特定名词的检索效果好。然而，其主要缺点是无法理解语义，例如它无法识别“汽车”和“轿车”是同义词，存在“词汇鸿沟”。

稀疏向量的核心思想是只存储非零值。例如，一个8维的向量`[0,0,0,5,0,0,0,9]`，其大部分元素都是零。用稀疏格式表示，可以极大地节约空间。常见的稀疏表示法有两种：

1. 字典/键值对：这种方式将非零元素的`索引`作为键，`值`作为值。上面的向量可以表示为：

   ```python
   // {索引: 值}
   {
     "3": 5,
     "7": 9
   }
   ```

2. 坐标列表：这种方式通常用一个元组`(维度, [索引列表], [值列表])`来表示。上面的向量可以表示为：

   ```python
   (8, [3, 7], [5, 9])
   ```

   这种格式在`Scipy`等科学库中非常常见。

假设在一个包含5万个词的词汇表中，“西红柿”在第88位，”炒“在第666位，”蛋“在第999位，它们的BM25权重分别是1.2、0.8、1.5。那么它的稀疏表示就是：

```python
// {索引: 权重}
{
  "88": 1.2,
  "666": 0.8,
  "999": 1.5
}
```



**稠密向量**也常被称为“语义向量”，是通过深度学习模型学习到的数据（如文本、图像）的低维、稠密的浮点数表示。这些向量旨在将原始数据映射到一个连续的、充满意义的“语义空间”中来捕捉“语义”或“概念”。在理想的语义空间中，向量之间的距离和方向代表了它们所表示概念之间的关系。一个经典的例子是`vector('国王') - vector('男人') + vector('女人')`的计算结果在向量空间中非常接近`vector('女王')`，这表明模型学会了“性别”和“皇室”这两个维度的抽象概念。它的经典代表包括Word2Vec、GloVe、以及所有基于Transformer的模型（如BERT、GPT）生成的嵌入（Embedding）。

其主要优点是能够理解同义词、近义词和上下文关系，泛化能力强，在语义搜索任务中表现卓越。但其缺点也同样明显：可解释性差（向量中的每个维度通常没有具体的意义），需要大量数据和算力进行模型训练，且对未登录词（OOV）[^1]的处理相对困难。

与稀疏向量不同，稠密向量的所有维度都有值，因此使用数组来表示。一个预训练好的语义模型在读取“西红柿炒蛋”后，会输出一个低维的稠密向量：

```python
// 这是一个低维（比如1024维）的浮点数向量
// 向量的每个维度没有直接的、可解释的含义
[0.89, -0.12, 0.77, ..., -0.45]
```

这个向量本身难以解读，但它在语义空间中的位置可能与“番茄鸡蛋面”、“洋葱炒鸡蛋”等菜肴的向量非常接近，因为模型理解了它们共享“鸡蛋类菜肴”、“家常菜”、“酸甜口味”等核心概念。因此，当我们搜索“蛋白质丰富的家常菜”时，即使查询中没有出现任何原文关键词，稠密向量也很有可能成功匹配到这份菜谱。

## 混合检索

通过上文可以看出稀疏向量和稠密向量各自的特点，如果我们将它们结合起来，优势互补，就成了一个不错的选择。混合检索就是基于这个思路。通过结合多种搜索算法（最常见的是稀疏和稠密检索）来提升搜索结果相关性和召回率。

### 技术原理与融合方法

混合检索通常并行执行两种检索算法，然后将两组异构的结果集融合成一个统一的排序列表。

1. 倒数排序融合（Reciprocal Rank Fusion, RRF）

   RRF不关心不同检索系统的原始得分，只关心每个文档在各自结果集中的排名。其思想是：一个文档在不同检索系统重的排名越靠前，它的最终得分越高。

   其计分公式为：

   $$ RRF_{score} (d) = \sum _{i=1} ^{k} \frac{1} {rank _{i} (d) + c}$$

   其中：

   - d是待评分的文档。
   - k是检索系统的数量（这里是2，即稀疏和稠密）。
   - $rank_i(d)$是文档d在第i个检索系统中的排名。
   - c是一个常数（通常为60），用于降低排名靠后文档的权重，避免它们对结果产生过大影响。

2. 加权线性组合

   这种方法需要先将不同检索系统的得分进行归一化（例如，统一到0-1区间），然后通过一个权重参数$\alpha$来进行线性组合。

    $$Hybrid_{score}=\alpha\cdot Dense_{score}+(1-\alpha)\cdot Sparse_{score}$$

   通过调整$\alpha$的值，可以灵活地控制语义相关性与关键词匹配在最终排序中的贡献比例。例如，在电商搜索中可以调高关键词的权重；而在智能问答中，则侧重于语义。

### 优势与局限

| 优势 | 局限 |
| :--- | :--- |
| **召回率与准确率高**：能同时捕获关键词和语义，显著优于单一检索。 | **计算资源消耗大**：需要同时维护和查询两套索引。 |
| **灵活性强**：可通过融合策略和权重调整，适应不同业务场景。 | **参数调试复杂**：融合权重等超参数需要反复实验调优。 |
| **容错性好**：关键词检索可部分弥补向量模型对拼写错误或罕见词的敏感性。 | **可解释性仍是挑战**：融合后的结果排序理由难以直观分析。 |

## 代码实践

```python
import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, AaaSearchRequest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 1. 初始化设置
COLLECTION_NAME = "dragon_hybrid_demo"
MILVUS_URI = "http://localhost:19530"  # 服务器模式
DATA_PATH = "../../data/C4/metadata/dragon.json"  # 相对路径
BATCH_SIZE = 50

# 2. 连接Milvus并初始化嵌入模型
print(f"正在连接到Milvus：{MILVUS_URI}")
connections.connect(uri=MILVUS_URI)

print("--> 正在初始化 BGE-M3 嵌入模型...")
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
print(f"--> 嵌入模型初始化完成。密集向量维度: {ef.dim['dense']}")

# 3. 创建Collection
milvus_client = MilvusClient(uri=MILVUS_URI)
if milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在删除已存在的 Collection '{COLLECTION_NAME}'...")
    milvus_client.drop_collection(COLLECTION_NAME)
    
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"])
]

# 如果集合不存在，测创建它及索引
if not milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在创建 Collection '{COLLECTION_NAME}'...")
    schema = CollectionSchema(fields, description="关于龙的混合检索示例")
    # 创建集合
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")
    print("--> Collection 创建成功。")

    # 创建索引
    print("--> 正在为新集合创建索引...")
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    collection.create_index("sparse_vector", sparse_index)
    print("稀疏向量索引创建成功。")

    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    collection.create_index("dense_vector", dense_index)
    print("密集向量索引创建成功。")

collection = Collection(COLLECTION_NAME)
collection.load()
print(f"--> Collection '{COLLECTION_NAME}' 已加载到内存。")
```



[^1]:




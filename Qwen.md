# Qwen

![框架图](assets/framework.jpeg)

GPT这类模型都是由解码器为主要结构的模型，归根结底这类模型需要不断地自回归预测下一个 token，所以解码器是必不可少的一个结构。

上图即为Qwen2的主要结构，其中：

- `Tokenizer`是将文本转换为词表的工具，一般由一个预训练的model提供分词能力，例如Qwen2.5采用了BBPE提供分词
- `Embedding`是将分词转换为词向量
- `Attention_mask`是用来设定掩码遮蔽左边、右边或双向的
- 各类下游任务，`Casual`,`seqcls`等，基本都是基础模型`model`后面接对应的`Linear`层，还有损失函数不一样。

## 1. Qwen2Config

Qwen2Config中包含一些自定义的超参数，例如`vocab_size`,`hidden_size`,`num_hidden_layers`, `num_attention_heads`等。类似于`dict`可以调用里面的超参数:`config.pad_token_id`。

### 1.1 Qwen2Model

#### 1.1.1 初始化

- 设置了模型的两个属性：`padding_idx`（用于指定填充标记的索引），`vocab_size`（词汇表的大小）
- 初始化了模型的嵌入层、解码器层、归一化层
- 设置了是否使用`gradient_checkpoint`主要是用来节省显存
- 调用`post_init()`完成一些初始化和准备检查的工作

```python
class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(slef,config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        # Inintialize weights and apply final processing
        self.post_init()
```

其中`post_init()`主要是对参数初始化以及初始化梯度检查点：

```python
def post_init(self):
    self.init_weights()
    self._backward_compatibility_gradient_checkpointing()
```

#### 1.1.2 Forward


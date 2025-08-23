# GraphRAG

> 本文档参考了[DataWhale](https://github.com/datawhalechina/tiny-universe/tree/main/content/TinyGraphRAG)，感谢！

## 项目动机

RAG技术在分块策略上存在一些问题，例如：

```
chunk 1
小明的爷爷叫老明

chunk 2
小明的爷爷是一个木匠

chunk 3
小明的爷爷...
```

假如我们希望查询的信息是：“小明认识的木匠叫什么名字？”，很明显这段话需要召回两个chunk来回答。显而易见，chunk2的相关性更强，但是仅仅召回chunk2并不能回答这个问题。而真正关键的chunk1片段可能无法碑召回，原因在于，**分块策略实际上破坏了文档的语义连续性。**有些工作，比如**late chunking策略**和**GraphRAG**都是一些解决这个问题的方案。

同时，RAG技术还面临一个比较棘手的问题，即全局信息的查询。假如我们有一个文档，我们希望查询“文档中小明所有家人的信息”，这同样是一个重大挑战。因为这些信息可能存在不同的位置，按照分块检索的难度很大。同时，还有可能涉及一些复杂的推理问题，同样也有一些工作提出了Agentic RAG策略来尝试解决，GraphRAG也提供了自己的解决方案，即通过图中的社区聚类，预先聚类信息用以应对用户的提问。

**值得注意的是，去阅读和复现源码仍然十分的重要！**

## 前置实现

### 1. 实现LLM模块

首先我们需要实现LLM模块，这是系统中最基本的模块，我们将利用大模型完成文档的清洗、信息提取等工作，可以说GraphRAG的一部分精髓就是使用大模型预先处理文档信息，方便后续检索。

```python
from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseLLM(ABC):
    '''Interface for large language models.
    
    Args: 
    	model_name(str): The name of LLM.
    	model_params(Optional[dict[str, Any]], optional): Additional parameters passed to the model when text is sent to it. Defaults to None.
    	**kwargs(Any): Arguments passed to the model when for the class is initialised. Defualts to None.
    '''
    def __init__(
        self, 
        model_name:str, 
        model_params: Optional[dict[str, Any]] = None, 
        **kwargs: Any
    ):
        self.model_name = model_name
        self.model_params = model_params
        
    @abstractmethod
    def predict(self, input: str) -> str:
        '''Sends a text input to the LLM and retrieves a response.
        
        Args:
        	input(str): Text sent to the LLM.
        	
        Returns:
        	str: The response from the LLM.
        '''
        pass
```

如上是一个调用大模型的抽象接口，这可以帮助我们统一大模型的调用形式，继承这个类，就可以实现不同模型的调用接口。

```python
from zhipuai import ZhipuAI
from typing import Any, Optional
from .base import BaseLLM

class zhipuLLM(BaseLLM):
    '''Implementation of the BaseLLm interface using ZhipuAI.'''
    def __init__(
        self, 
        model_name: str, 
        api_key: str, 
        model_params: Optional[dict[str, Any]] = None, 
        **kwargs: Any
    ):
        super().__init__(model_name, model_params, **kwargs)
        self.client = ZhipuAI(api_key=api_key)
        
    def predict(self, input: str) -> str:
        '''Sends a text input to the ZhipuAI model and retrieves a response.
        
        Args:
            input(str): Text sent to the ZhipuAI model.
            
        Returns:
            str: The response from the ZhipuAI model.
        '''
        response = self.client.chat.completions.create(
            model=self.model_name, 
            messages=[{'role':'user', 'content':input}])
        return response.choices[0].message.content
```

完成上面的代码之后，我们可以尝试调用`predict`方法测试一下。

```python
llm = zhipuLLM(model_name='...', api_key='...')
print(llm.predict('Hello, how are you?'))
```

当观察到LLM返回正确的回复之后，这个模块就构建完成了。

### 2. 实现Embedding模块

除了调用大模型，我们还需要实现Embedding模块，Embedding模块用于将文本转换为向量，我们将使用向量来表示文档中的信息，这样的好处是：我们可以通过向量的相似度来衡量文档与查询之间的相似度，从而召回对回复用户问题最有帮助的文档。

```python
from abc import ABC, abstractmethod
from typing import List, Any, Optional

class BaseEmb(ABC):
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
    	**kwargs: Any
    ):
        self.model_name = model_name
        self.model_params = model_params or {}
       
    @abstractmethod
    def get_emb(self, input: str) -> List[float]:
        '''Sends a text input to the embedding model and retrieves the embedding.
        
        Args:
        	input(str): Text sent to the embedding model.
        	
        Returns:
        	List[float]: The embedding vector from the model.
        '''
        pass
```

```python
from zhipuai import ZhipuAI
from typing import List
from .base import BaseEmb

class zhipuEmb(BaseEmb):
    def __init__(
        self, 
        model_name: str, 
        model_params: Optional[dict[str, Any]] = None,
        api_key: str,
        **kwargs
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.client = ZhipuAI(api_key=api_key)
        
    def get_emb(self, text: str) -> List[float]:
        emb = self.client.embeddings.create(
        	model=self.model_name,
        	input=text
        )
        return emb
```

完成之后，可以尝试调用`get_emb`方法来测试是否成功。

```python
emb = zhipuEmb(model_name='...', api_key='...')
print(emb.get_emb('Hello, how are you?'))
```

当观察到Embedding模块正确给出编码后的向量，说明这个模块就构建完成了。

### 3. 实现与Neo4j的交互

我们需要准备一个图数据库，用以进行图的存储和查询，以及一些必要的图操作算法，我们选择使用`Neo4j`作为图数据库。

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
	url,
    auth=(username, password)
	)
```

完成以上设置后，我们就可以采用下面的方法来运行Cypher查询语句了。Cypher查询语句是Neo4j的查询语言，类似于SQL语言。

```python
with dirver.session() as session:
    result = session.run('MATCH (n) RETURN n') # 查询图中所有的节点
    for record in result:
        print(record)
```

## 核心实现


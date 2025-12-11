
# FlashAttention
## 一、NVIDIA GPU编程架构及内存架构
### 1. 编程架构
一个典型的NVIDIA GPU线程架构层级可以表示为：
**Kernel->Grids->Blocks->Tread**
其结构如下图所示：

![GPU编程架构1](assets/GPU%E7%BC%96%E7%A8%8B%E6%9E%B6%E6%9E%841.png)

![GPU编程架构2](assets/GPU%E7%BC%96%E7%A8%8B%E6%9E%B6%E6%9E%842.png)

- Kernel：Kernel是GPU上执行的核心程序，这个Kernel对应一个Grid
- Grid：Grid是最高级别的并行计算单位，由多个线程块（Block）组成，可同时在GPU上执行
- Block：线程块Block是网格的组成部分，包含多个线程，可协作和共享资源
- Thread：Thread是最基本的并行执行单元，存在于线程块内部，每个线程可以独立执行计算任务，根据自身的线程索引ID执行
### 2. 内存架构
NVIDIA GPU中的不同类型的内存有不同的速度、大小及访问限制。GPU显存分为**全局内存（Global Memory）**、**本地内存（Local Memory）**、**共享存储（Shared Memory，SRAM）**、**寄存器（Register）**、**常量内存（Constant Memory）**、**纹理内存（Texture Memory）** 六大类。其结构如下图所示：

![GPU内存架构](assets/GPU%E5%86%85%E5%AD%98%E6%9E%B6%E6%9E%84.png)

- 全局内存、本地内存、共享内存和寄存器具有读写能力。
- 全局内存和本地内存使用高带宽显存（High Bandwidth Memory）位于板卡RAM存储芯片上，该部分内存容量很大。所有线程都可以访问全局内存，而本地内存只能由当前线程访问。NVIDIA H100中全局内存有80GB空间，其访问速度虽然可以达到3.35TB/s，但当全部线程同时访问全局内存时，其平均带宽仍然很低。
- 共享存储和寄存器位于GPU芯片上，因此容量很小，并且只有在同一个GPU线程块（Thread Block）内的线程才可以并行访问共享存储，而寄存器仅限于同一个线程内部访问。虽然NVIDIA H100中每个GPU线程块在流式多处理器（Stream Multi-processor）上可以使用的共享存储容量仅有228KB，但是其速度比全局内存的访问速度快很多。
## 二、FlashAttention
### 1. Self-Attention
在self-attention操作中，传统的方法需要引入两个中间矩阵S和P，并存储倒全局内存中。具体的计算过程如下：
$$S=QK,\quad P=Softmax(S),\quad O=PV$$
按照上述计算过程，需要先从全局内存中读取矩阵Q和K，并将计算好的矩阵S写入全局内存，然后从全局内存中获取矩阵S，计算Softmax得到矩阵P，再将其写入全局内存，最后读取矩阵P和矩阵V，计算得到矩阵O。这样的过程会极大地占用显存的带宽。在自注意力机制中，GPU的计算速度比内存速度快得多，因此计算效率越来越受全局内存访问制约。
![[传统自注意力机制矩阵.png]]

### 2. FlashAttention V1

FlashAttention利用GPU硬件中的特殊设计，针对全局内存和共享存储的IO速度不同，尽可能地避免从HBM中读取或写入注意力矩阵。达成该目标需要做到在不访问整个输入的情况下计算Softmax函数，并且后向传播中不能存储中间注意力矩阵。在标准Attention算法中，Softmax计算按行进行，即在与V做矩阵乘法之前，需要完成Q、K每个分块中的一整行的计算。在得到Softmax的结果后，再与矩阵V分块做矩阵乘。而在FlashAttention中，将输入分割成块，并在输入块上进行多次传递，以增量方式执行Softmax计算。
自注意力算法的标准实现将计算过程中的矩阵S、P写入全局内存，而**这些中间矩阵的大小与输入的序列长度有关且为二次型**。因此，FlashAttention就提出了不使用中间自注意力矩阵，通过存储归一化因子来减少全局内存消耗的方法。FlashAttention算法并没有将S、P整体写入全局内存，而是通过分块写入，存储前向传播的Softmax归一化因子，在后向传播中快速重新计算片上注意力，这比从全局内存中读取中间注意力矩阵的标准方法更快。虽然大幅减少了全局内存的访问量，重新计算也导致了FLOPS增加，但其运行的速度更快且使用的内存更少。具体算法如下：

### 3. FlashAttention V2

FlashAttention2在FlashAttention1的基础上做了以下优化：

- 减少非矩阵乘法运算：在FlashAttention1中，整个计算的流程是以Q的分块为外层循环的。这意味着对于Q的每个块，我们都需要遍历整个K和V。在计算Softmax时，需要进行一些规约操作来保证数值稳定性。这些规约操作虽然计算量不大，但是次数非常多，在GPU上计算效率低。因此**FlashAttention2将外层循环改为在K和V的分块上进行**。这样，对于一次加载的K和V的分块，我们可以一次性计算Q的多个块的注意力分数，显著减少了Softmax规约计算的次数。
- 更合理的并行化方案：FlashAttention1的并行化主要集中在**批量大小**和**注意力头**这两个维度。当这两个维度很小时，可用的并行度就很低。FlashAttention2增加了在**序列长度**维度上的并行化。因为FlashAttention2的外层循环是K和V的分块，那么不同的Q分块之间的计算就是相互独立的，因此FlashAttention2将不同的Q分块分配给GPU上不同的线程块同时进行计算。
- 更高效的任务调度：FlashAttention2重新设计了内核任务调度，确保了Warp独立性并且使用了更高效的通信原语，减少了通信开销，提高了通信效率。

![FlashAttention算法步骤](assets/FlashAttention%E7%AE%97%E6%B3%95%E6%AD%A5%E9%AA%A4.png)

![FlashAttention计算流程图](assets/FlashAttention%E8%AE%A1%E7%AE%97%E6%B5%81%E7%A8%8B%E5%9B%BE.png)




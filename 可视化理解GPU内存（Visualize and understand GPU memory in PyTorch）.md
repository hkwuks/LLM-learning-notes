# Visualize and understand GPU memory in PyTorch

## The PyTorch visualizer

```python
import torch
from torch import nn

# Start recording memory snapshot history
torch.cuda.memory._record_memory_history(max_entries=100000)

model = nn.Linear(10000,50000,device='cuda')

for _ in range(3):
inputs = torch.randn(5000,10000,device='cuda')
outputs = model(inputs)

# Dump memory snapshot history to a file and stop recording
torch.cuda.memory._dump_snapshot('profile.pkl')
torch.cuda.memory._record_memory_history(enabled=None)
```

Running this code generates a `profile.pkl` file that contains a history of GPU memory usage during execution. You can visualize this history at: [https://pytorch.org/memory_viz](https://pytorch.org/memory_viz).

By dragging and dropping your `profile.pkl` file, you will see a graph like this:

![Pasted image 20250724213653.png](assets/Pasted image 20250724213653.png)
 Let's break down this graph into key parts:
   ![1753450394426.jpg](assets/1753450394426_1753450400053_0.jpg)

1. **Model Creation**: Memory increases by 2 GB, corresponding to the model's size:

   $10000 × 50000\; weights + 5000\; biases\; in\; float32\; (4\; bytes) = (5 × 10^8) × 4\; bytes = 2\; GB$

   This memory ==(in blue)== persists throughout execution.

2. **Input Tensor Creation (1st Loop)**: Memory increases by 200 MB matching the input tensor size:

   $5000\; \times 1000\; elements\; in\; float32\; (4\; bytes) = (5 \times 10^{7}) \times 4\; bytes = 0.2 GB$

3. **Forward Pass (1st Loop)**: Memory increases by 1GB for the output tensor:

   $5000\; \times 50000\; elements\; in\; float32\; (4\; bytes) = (25 \times 10^{7}) \times 4\; bytes = 1 GB$

4. **Input Tensor Creation (2nd Loop)**: Memory increases by 200 MB for a new input tensor. At this point, you might expect the input tensor from step 2 to be freed. Still, it isn’t: ==the model retains its activation, so even if the tensor is no longer assigned to the variable `inputs`, it remains referenced by the model’s forward pass computation.== The model retains its activations because these tensors are required for the backpropagation process in neural networks. Try with `torch.no_grad()` or `torch.inference_mode()` to see the difference.
5. **Forward Pass (2nd Loop)**: Memory increases by 1 GB for the new output tensor, calculated as in step 3.
6. **Release 1st Loop Activation**: After the second loop’s forward pass, the input tensor from the first loop (step 2) can be freed. The model’s activations, which hold the first input tensor, are overwritten by the second loop’s input. Once the second loop completes, the first tensor is no longer referenced and its memory can be released.
7. **Update `output`**: The output tensor from step 3 is reassigned to the variable `output`. The previous tensor is no longer referenced and is deleted, freeing its memory.
8. **Input Tensor Creation (3rd Loop)**: Same as step 4.
9. **Forward Pass (3rd Loop)**: Same as step 5.
10. **Release 2nd Loop Activation**: The input tensor from step 4 is freed.
11. **Update `output` Again**: The output tensor from step 5 is reassigned to the variable `output`, freeing the previous tensor.
12. **End of Code Execution**: All memory is released.



## Visualizing Memory During Training

The previous example was simplified. In real scenarios, we often train complex models rather than a single linear layer. Additionally, the earlier example did not include the training process. Here, we will examine how GPU memory behaves during a complete training loop for a real large language model.




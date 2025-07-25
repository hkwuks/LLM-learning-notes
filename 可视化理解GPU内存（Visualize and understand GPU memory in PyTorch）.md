## The PyTorch visualizer

```python
import torch
from torch import nn

# Start recording memory snapshot history
torch.cuda.memory._record_memory_history(max_entries=100000)

model = nn.Linear(10000,50000,device='cuda')

for _ in range(3):
inputs = torch.randn(50000,10000,device='cuda')
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

- **Model Creation**: Memory increases by 2GB, corresponding to the model's size:
  logseq.order-list-type:: number
  - $10000 × 50000\, weights + 5000\, biases\, in\, float32\, (4\, bytes) = (5 × 10^8) × 4\, bytes = 2\, GB$
  - This memory ==(in blue)== persists throughout execution.

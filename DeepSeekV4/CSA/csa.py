import torch

class DeepseekV4HCACache(DynamicSlidingWindowLayer):
    layer_type = "heavily_compressed_attention"

    def __init__(self, config):
        super().__init__(config)
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.buffer_kv: dict[str, torch.Tensor | None] = {"compressor":None}
        self.buffer_gate: dict[str, torch.Tensor | None] = {"compressor": None}
        self.compressed_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.entry_count: dict[str, int] = {"compressor": 0}

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        '''
        Shared sliding-window K=V update body. V4 uses shared-KV MQA, so `keys` and `values` point to the same storage on every layer.
        '''
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.values = self.keys
        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)
        self.keys = full[:, :, -self.sliding_windows + 1:, :]
        self.values = self.keys
        return full, full
    
    def store_compression_weights(self, name: str, kv: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        

class DeepseekV4CSACache(DeepseekV4HCACache):
    layer_type = "compressed_sparse_attention"

    def __init__(self, config):
        super().__init__(config)
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.buffer_kv["indexer"] = None
        self.buffer_gate["indexer"] = None
        self.compress_kv["indexer"] = None
        self.entry_count["indexer"] = 0
        self.overlap_kv = dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
        self.overlap_gate: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}

    def update_overlap_state(self, name: str, chunk_kv: torch.Tensor, chunk_gate: torch.Tensor, head_dim: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        '''
        Read the 'name' entry's prior window's Ca slice (saved on the previous forward call) and persist the *current* call's last-window Ca slice for the next call. Only the `"head_dim` slice (Ca) is ever comsumed downstream - Cb has already been folded into the previous window's emitted compressed entry - so we store half what `chunk[:, -1]` holds. 
        Returns `(prior_kv, prior_gate)` - both `None` on the very first call.
        '''

        prior_kv, prior_gate = self.overlap_kv[name], self.overlap_gate[name]
        self.overlap_kv[name] = chunk_kv[:, -1, :, head_dim].clone()
        self.overlap_gate[name] = chunk_gate[:, -1, :, head_dim].clone()
        return prior_kv, prior_gate
    

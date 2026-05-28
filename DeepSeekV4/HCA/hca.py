import torch
import torch.nn as nn

class DynamicSlidingWindowLayer(DynamicLayer):
    is_sliding = True

    def __init__(self, config: PreTrainedConfig | None, sliding_window: int | None = None):
        super().__init__()
        if sliding_window is None:
            if config is None:
                raise ValueError("Either `config` or `sliding_window` must be provided.")
            sliding_window = getattr(config, "sliding_window", None) or getattr(config, "attention_chunk_size", None)
        self.sliding_window = sliding_window
        self.cumulative_length = 0
        self._sliding_window_tensor = torch.tensor(self.sliding_window, dtype=torch.long)
    
    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        super().lazy_initialization(key_states, value_states)
        self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
    
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.cumulative_length += key_states.shape[-2]

        full_key_states = torch.cat([self.keys, key_states], dim=-2)
        full_value_states = torch.cat([self.values, value_states], dim=-2)
        self.keys = full_key_states[:, :, -self.sliding_window + 1:, :]
        self.values = full_value_states[:, :, -self.sliding_window + 1:, :]

        return full_key_states, full_value_states
    
    def get_mask_sizes(self, query_length: int)-> tuple[int, int]:
        is_full = self.cumulative_length >= self.sliding_window
        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        if is_full:
            kv_length = self.sliding_window - 1 + query_length
        else:
            kv_length = self.cumulative_length + query_length
        return kv_length, kv_offset
    
    def get_seq_length(self) -> int:
        return self.cumulative_length
    
    def get_max_cache_shape(self)-> int:
        return self.sliding_window

    def crop(self, max_length: int)->None:
        if self.get_seq_length() >= self.sliding_window:
            raise ValueError("Cannot `crop` a `DynamicSlidingWindowLayer` after it has seen more tokens than its sliding window (otherwise some states are lost)")
        super().crop(max_length)
        self.cumulative_length = self.keys.shape[-2]


class DeepseekV4HCACache(DynamicSlidingWindowLayer):
    layer_type = "heavily_compressed_attention"

    def __init__(self, config: "DeepseekV4Config"):
        super().__init__(config)
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.buffer_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.buffer_gate: dict[str, torch.Tensor | None] = {"compressor": None}
        self.compressed_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.entry_count: dict[str, int] = {"compressor": 0}

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.values = self.keys

        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)
        self.keys = full[:, :, -self.sliding_window + 1:, :]
        self.values = self.keys
        return full, full
    
    def store_compression_weights(self, name: str, kv: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        first_window_position = self.entry_count[name] * self.compress_rate
        buffered_kv, buffered_gate = self.buffer_kv[name], self.buffer_gate[name]
        if buffered_kv is not None and buffered_kv.shape[1]:
            kv = torch.cat([buffered_kv, kv], dim=1)
            gate = torch.cat([buffered_gate, gate], dim=1)
        # only return the longest prefix that is a multiple of `compress_rate`; the rest stays in the buffer for next time
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        self.buffer_kv[name], self.buffer_gate[name] = kv[:, usable:], gate[:, usable:]
        return kv[:, :usable], gate[:, :usable], first_window_position
    
    def update_compressed_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        if self.compressed_kv[name] is None:
            self.compressed_kv[name] = compressed
        elif compressed.shape[1] > 0:
            self.compressed_kv[name] = torch.cat([self.compressed_kv[name], compressed], dim=1)
        self.entry_count[name] += compressed.shape[1]
        return self.compressed_kv[name]
    
class DeepseekV4HCACompressor(nn.Module):
    rope_layer_type = 'compress'

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(self, hidden_states: torch.Tensor, q_residual: torch.Tensor, position_ids: torch.Tensor, past_key_values: Cache | None, layer_idx: int) -> tuple(torch.Tensor, torch.Tensor):
        batch, _, _ = hidden_states.shape
        cache_layer: DeepseekV4HCACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)
        
        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias.to(chunk_gate.dtype)
            compressed = self.kv_norm((chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2))
            positions = torch.arange(n_windows, device=compressed.device)
            positions = (positions * self.compress_rate + first_window_position).unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros(batch, 0, self.head_dim)
        
        if cache_layer is not None:
            compressed = cache_layer.update_compressed_states("compressor", compressed)
        compressed_kv = compressed.unsqueeze(1)

        compressed_len = compressed_kv.shape[2]
        seq_len = position_ids.shape[1]
        if seq_len == 1 or compressed_len == 0:
            return compressed_kv, None
        
        # query `t` may only see cache entries at pos `w` t > w * compress_rate (ex: t=7, w=2 does not attend to it).
        entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
        causal_threshold = (position_ids + 1) // self.compress_rate # [B, S]
        block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
        block_bias = block_bias.masked_fill(entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1), float("-inf"))
        return compressed_kv, block_bias
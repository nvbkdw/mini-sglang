import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

from nano.models.config import ModelConfig, RotaryConfig
from nano.models.ops.rotary import get_rope


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        from flashinfer import fused_add_rmsnorm, rmsnorm
        
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps
        
        self.fused_add_rmsnorm = fused_add_rmsnorm
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x.view(-1, x.shape[-1]), residual.view(-1, residual.shape[-1]), self.weight, self.eps)
        return x, residual


class Attention(nn.Module):
    def __init__(self, 
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_config = rotary_config
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = qkv.shape[:2]
        q, k, v = qkv.split([self.num_qo_heads * self.head_dim, self.num_kv_heads * self.head_dim, self.num_kv_heads * self.head_dim], dim=-1)
        
        # Reshape to [seq_len, num_heads, head_dim] for RoPE and norms
        q = q.view(-1, self.num_qo_heads, self.head_dim).contiguous()
        k = k.view(-1, self.num_kv_heads, self.head_dim).contiguous()
        v = v.view(-1, self.num_kv_heads, self.head_dim).contiguous()
        
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q)
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k)
        
        # Apply RoPE inplace - expects [num_tokens, num_heads, head_dim]
        positions = torch.arange(seq_len, dtype=torch.int32, device=q.device).repeat(batch_size, 1) # [batch_size, seq_len]
        self.rotary(positions, q, k)
        
        # TODO: use optimized attention kernel, i.e. flash_infer
        # Reshape for scaled_dot_product_attention: [batch, num_heads, seq_len, head_dim]
        q = q.view(-1, seq_len, self.num_qo_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(-1, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(-1, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        
        # Reshape back to [seq_len, num_qo_heads * head_dim]
        o = o.permute(0, 2, 1, 3).reshape(-1, seq_len, self.num_qo_heads * self.head_dim)
        return o
    

        
class Qwen3Attn(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int, backend: str = "PyTorchBackend", *, has_attn_bias: bool = False, has_qk_norm: bool = False):
        super().__init__()
        GQA_ratio = config.num_qo_heads // config.num_kv_heads
        output_size = (GQA_ratio + 2) * config.num_kv_heads * config.head_dim
        self.backend = backend
        self.qkv_proj = nn.Linear(config.hidden_size, output_size, bias=False)
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        
        # TODO: integrate attention kernel
        if backend == "FlashInferBackend":
            from nano.models.ops.attention import FlashInferBackend
            self.attn = FlashInferBackend(config, layer_id)
        else:
            self.attn = Attention(
                num_qo_heads=config.num_qo_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                rotary_config=config.rotary_config,
                q_norm=self.q_norm,
                k_norm=self.k_norm,
            )
        self.o_proj = nn.Linear(config.num_qo_heads * config.head_dim, config.hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(x)
        del x
        o = self.attn(qkv)
        return self.o_proj(o)

        
class Qwen3MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # output size is 2x intermediate size to match the output of the gate_up_proj
        # combines two linear layer into one
        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size*2, bias=False)
        
        # Fused SiLU and Mul operation.
        # silu(input[..., :hidden_size]) * input[..., hidden_size:]
        from flashinfer import silu_and_mul
        self.act_fn = silu_and_mul
        
        # output linear layer
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj(y)
        

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()
        self._layer_id = layer_id
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        x, residual = self.input_layernorm(x, residual)
        with nvtx.range(f"MHA_{self._layer_id}"):
            x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm(x, residual)
        with nvtx.range(f"MLP_{self._layer_id}"):
            x = self.mlp(x)
        return x, residual

class Qwen3Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)])
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with nvtx.range("Embedding"):
            x = self.embed_tokens(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers:
            with nvtx.range(f"Layer_{layer._layer_id}"):
                x, residual = layer(x, residual)
        return self.norm(x, residual)[0]

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model = Qwen3Model(config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.model.forward(input_ids)
        # lm_head
        with nvtx.range("LMHead"):
            logits = F.linear(x, self.model.embed_tokens.weight, None)
            
        return logits
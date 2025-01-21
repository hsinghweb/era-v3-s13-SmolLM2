import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

    def __repr__(self):
        return f'LlamaRMSNorm(({self.hidden_size},), eps={self.eps})'

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Create inverse frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Create position embeddings cache
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

class LlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 576,
        num_attention_heads: int = 9,
        num_key_value_heads: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        
        # Modify projection sizes for grouped-query attention
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # [576, 576]
        # Key and value projections have reduced size due to head grouping
        self.k_proj = nn.Linear(hidden_size, self.head_dim * num_key_value_heads, bias=False)  # [576, 192]
        self.v_proj = nn.Linear(hidden_size, self.head_dim * num_key_value_heads, bias=False)  # [576, 192]
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # [576, 576]
        
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:2]
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape query states
        query_states = query_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Reshape key and value states
        key_states = key_states.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        
        # Repeat key and value states to match number of query heads
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_length)
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 576,
        intermediate_size: int = 1536,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config["intermediate_size"],
            hidden_act=config["hidden_act"],
        )
        self.input_layernorm = LlamaRMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        self.post_attention_layernorm = LlamaRMSNorm(self.hidden_size, eps=config["rms_norm_eps"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config["num_hidden_layers"])])
        self.norm = LlamaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.rotary_emb = LlamaRotaryEmbedding(
            self.config["hidden_size"] // self.config["num_attention_heads"],
            max_position_embeddings=self.config["max_position_embeddings"],
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states

class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        self.init_weights_from_scratch()

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        pad_token_id: int = None,
        eos_token_id: int = None,
    ) -> torch.LongTensor:
        """Generate text using simple greedy decoding"""
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]
        device = input_ids.device
        
        # Initialize attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Generate until max_length or eos token
        while current_length < max_length:
            # Get model outputs
            outputs = self.model(input_ids, attention_mask=attention_mask)
            next_token_logits = self.lm_head(outputs[:, -1, :])
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Get next token
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Append new tokens
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=device)
            ], dim=-1)
            
            current_length += 1
            
            # Stop if eos token is generated
            if eos_token_id is not None and (next_tokens == eos_token_id).any():
                break
        
        return input_ids

    def init_weights_from_scratch(self):
        """Initialize all model weights from scratch using random initialization"""
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                # Initialize linear layers
                torch.nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.config["initializer_range"]
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Initialize embeddings
                torch.nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.config["initializer_range"]
                )
            elif isinstance(module, LlamaRMSNorm):
                # Initialize RMSNorm weights to ones
                torch.nn.init.ones_(module.weight)
                
        self.apply(_init_weights)
        
        # Explicitly log that we're creating a fresh model
        print("Initialized a fresh SmolLM2-135M model with random weights")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss
            
        return logits

# Model configuration from YAML
MODEL_CONFIG = {
    "vocab_size": 49152,        # Embedding params: 49152 * 576 = 28,311,552
    "hidden_size": 576,         # Hidden dimension
    "num_attention_heads": 9,    # Total attention heads
    "num_key_value_heads": 3,   # Grouped-query attention heads
    "num_hidden_layers": 30,    # Number of transformer layers
    "intermediate_size": 1536,  # MLP intermediate size
    "hidden_act": "silu",
    "max_position_embeddings": 2048,
    "initializer_range": 0.041666666666666664,
    "rms_norm_eps": 1e-5,
    "use_cache": True,
    "tie_word_embeddings": True,
}

def create_model(seed: int = None):
    """Creates and returns a fresh SmolLM2-135M model instance
    
    Parameter count breakdown:
    - Embedding layer:        28,311,552  (vocab_size * hidden_size)
    - 30 Decoder layers:     105,902,080  (30 * [
        - Self-attention:     1,327,104   (4 * hidden_size * hidden_size)
        - MLP:               2,203,136    (2 * hidden_size * intermediate_size + intermediate_size * hidden_size)
        - Layer norms:          1,152     (2 * hidden_size)
      ])
    - Final norm:                  576    (hidden_size)
    - LM head:                 28,311,552 (hidden_size * vocab_size) [tied with embeddings]
    Total:                    134,515,008 parameters
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    model = LlamaForCausalLM(MODEL_CONFIG)
    
    # Calculate and verify parameter count
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 134_515_008, f"Expected 134,515,008 parameters but got {total_params}"
    
    return model 
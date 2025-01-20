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

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

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
        self.head_dim = hidden_size // num_attention_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.head_dim * num_key_value_heads, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.head_dim * num_key_value_heads, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
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
        
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        
        if attention_mask is not None:
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
        self.act_fn = F.silu

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
        self.config = config  # Store config for weight initialization
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        
        # Initialize all weights from scratch
        self.init_weights_from_scratch()
        
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
    "vocab_size": 49152,
    "hidden_size": 576,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "num_hidden_layers": 30,
    "intermediate_size": 1536,
    "hidden_act": "silu",
    "max_position_embeddings": 2048,
    "initializer_range": 0.041666666666666664,
    "rms_norm_eps": 1e-5,
    "use_cache": True,
    "tie_word_embeddings": True,
}

def create_model(seed: int = None):
    """Creates and returns a fresh SmolLM2-135M model instance with random initialization
    
    Args:
        seed (int, optional): Random seed for weight initialization
        
    Returns:
        LlamaForCausalLM: A freshly initialized model
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    model = LlamaForCausalLM(MODEL_CONFIG)
    
    # Calculate and print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created model with {total_params:,} parameters")
    
    return model 
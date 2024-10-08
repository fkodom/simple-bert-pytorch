from __future__ import annotations

from typing import Optional, Type, TypeVar, Union

import torch
from torch import Tensor, nn

from simple_bert_pytorch.common import Config


class Embeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        max_length: int = 512,
        pad_token_id: int = 0,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.1,
        type_vocab_size: int = 2,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_length, dim)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, dim)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        # These are purely for efficiency purposes. We could just as easily initialize
        # them in the forward pass, but we save some compute by doing it here.
        self.position_ids: Tensor
        self.register_buffer(
            "position_ids",
            torch.arange(max_length).expand((1, -1)),
            persistent=False,
        )
        self.token_type_ids: Tensor
        self.register_buffer(
            "token_type_ids",
            torch.zeros((1, max_length), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :seq_length]

        embeddings = self.word_embeddings.forward(input_ids)
        embeddings = embeddings + self.token_type_embeddings.forward(token_type_ids)
        embeddings = embeddings + self.position_embeddings.forward(position_ids)
        embeddings = self.LayerNorm.forward(embeddings)
        embeddings = self.dropout.forward(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of "
                f"attention heads ({num_heads})"
            )

        self.hidden_size = dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        num_heads = self.num_heads

        q = self.query.forward(hidden_states)
        k = self.key.forward(hidden_states)
        v = self.value.forward(hidden_states)

        q = q.view(batch_size, seq_len, num_heads, -1).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, num_heads, -1).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, num_heads, -1).permute(0, 2, 1, 3)

        # Input attention masks can have several shapes:
        #     (seq_len, ) - same mask applied to all inputs
        #     (batch_size, seq_len) - separate mask for each sample
        #     (batch_size, seq_len, seq_len) - separate masks for query and key/value
        #     (batch_size, num_heads, seq_len, seq_len) - separate masks for each head
        # But the input to scaled_dot_product_attention should have 4 dimensions.
        # We need to handle each use case here.
        if attention_mask is not None:
            if attention_mask.dim() == 1:
                # Add batch dimension
                attention_mask = attention_mask.unsqueeze(0)
            if attention_mask.dim() == 2:
                # Add query sequence dimension
                attention_mask = (
                    # Apply the same masking to query and key/value dimensions
                    attention_mask.unsqueeze(1)
                )
            if attention_mask.dim() == 3:
                # Add head dimension
                attention_mask = attention_mask.unsqueeze(1)

        dropout_p = self.dropout if self.training else 0.0
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout_p
        )

        return attn.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)


class SelfAttentionOutput(nn.Module):
    def __init__(self, dim: int, dropout: float, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.LayerNorm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense.forward(hidden_states)
        hidden_states = self.dropout.forward(hidden_states)
        hidden_states = self.LayerNorm.forward(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.self = SelfAttention(dim, num_heads=num_heads, dropout=dropout)
        self.output = SelfAttentionOutput(
            dim, dropout=dropout, layer_norm_eps=layer_norm_eps
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tensor:
        self_outputs = self.self.forward(hidden_states, attention_mask)
        return self.output.forward(self_outputs, hidden_states)


class Intermediate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: Union[str, nn.Module] = "gelu",
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(activation, str):
            activation = get_activation_fn(activation)
        self.activation = activation

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense.forward(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense.forward(hidden_states)
        hidden_states = self.dropout.forward(hidden_states)
        hidden_states = self.LayerNorm.forward(hidden_states + input_tensor)
        return hidden_states


class Layer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.attention = Attention(
            dim=dim, num_heads=num_heads, dropout=attention_dropout
        )
        self.intermediate = Intermediate(
            hidden_size=dim,
            intermediate_size=intermediate_size,
            activation=activation,
        )
        self.output = Output(
            intermediate_size=intermediate_size,
            hidden_size=dim,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tensor:
        attention = self.attention.forward(hidden_states, attention_mask)
        intermediate = self.intermediate.forward(attention)
        return self.output.forward(intermediate, attention)


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                Layer(
                    dim=dim,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        for layer in self.layer:
            hidden_states = layer.forward(hidden_states, attention_mask)
        return hidden_states


class Pooler(nn.Module):
    # NOTE: IMO, this is a model-specific layer.  It's relevant for (e.g.) classifiers
    # and embeddings models, some of those also use mean pooling instead.  The HF
    # implementation includes it in the BERT backbone, but I think it should be
    # withheld from the base model, and added to downstream models as needed.

    def __init__(self, dim: int):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        # For many common models, 'pooling" just means taking the first output token.
        x = hidden_states[:, 0]
        x = self.dense(x)
        return self.activation(x)


BackboneType = TypeVar("BackboneType", bound="Backbone")


class Backbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim: int,
        num_heads: int,
        intermediate_size: int,
        max_length: int = 512,
        pad_token_id: int = 0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = Embeddings(
            vocab_size=vocab_size,
            dim=dim,
            max_length=max_length,
            pad_token_id=pad_token_id,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
        )
        self.encoder = Encoder(
            num_layers=num_layers,
            dim=dim,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )

        # TODO: Initialize weights
        # self.post_init()

    @classmethod
    def from_config(cls: Type[BackboneType], config: Config) -> BackboneType:
        return cls(
            vocab_size=config["vocab_size"],
            num_layers=config["num_layers"],
            dim=config["dim"],
            num_heads=config["num_heads"],
            intermediate_size=config["intermediate_size"],
            max_length=config["max_length"],
            pad_token_id=config["pad_token_id"],
            dropout=config["dropout"],
            attention_dropout=config["attention_dropout"],
            activation=config["activation"],
            layer_norm_eps=config["layer_norm_eps"],
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        embeddings = self.embeddings.forward(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        return self.encoder.forward(embeddings, attention_mask=attention_mask)


def get_activation_fn(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "swish":
        return nn.SiLU()
    else:
        raise ValueError(f"Activation fn {name} not found")

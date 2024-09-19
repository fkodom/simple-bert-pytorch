from __future__ import annotations

import os
from enum import Enum
from typing import Dict, Union

import torch

from simple_bert_pytorch.modules import Backbone, Config, Pooler


class ModelName(str, Enum):
    BGE_SMALL_EN_V1_5 = "BAAI/bge-small-en-v1.5"
    BGE_BASE_EN_V1_5 = "BAAI/bge-base-en-v1.5"
    BGE_LARGE_EN_V1_5 = "BAAI/bge-large-en-v1.5"


class BGEConfig(Config):
    name: str
    weights_uri: str


CONFIGS: Dict[ModelName, BGEConfig] = {
    ModelName.BGE_SMALL_EN_V1_5: BGEConfig(
        name=ModelName.BGE_SMALL_EN_V1_5.value,
        weights_uri="https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/pytorch_model.bin",
        vocab_size=30522,
        num_layers=12,
        dim=384,
        num_heads=12,
        intermediate_size=1536,
        max_length=512,
        pad_token_id=0,
        dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-12,
    ),
    ModelName.BGE_BASE_EN_V1_5: BGEConfig(
        name=ModelName.BGE_BASE_EN_V1_5.value,
        weights_uri="https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/pytorch_model.bin",
        vocab_size=30522,
        num_layers=12,
        dim=768,
        num_heads=12,
        intermediate_size=3072,
        max_length=512,
        pad_token_id=0,
        dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-12,
    ),
    ModelName.BGE_LARGE_EN_V1_5: BGEConfig(
        name=ModelName.BGE_LARGE_EN_V1_5.value,
        weights_uri="https://huggingface.co/BAAI/bge-large-en-v1.5/resolve/main/pytorch_model.bin",
        vocab_size=30522,
        num_layers=24,
        dim=1024,
        num_heads=16,
        intermediate_size=4096,
        max_length=512,
        pad_token_id=0,
        dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-12,
    ),
}


class BGE(Backbone):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim: int,
        num_heads: int,
        intermediate_size: int,
        max_length: int,
        pad_token_id: int,
        dropout: float,
        attention_dropout: float,
        activation: str,
        layer_norm_eps: float,
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_layers=num_layers,
            dim=dim,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_length=max_length,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )
        self.pooler = Pooler(dim=dim)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        x = self.embeddings(input_ids)
        x = self.encoder(x, attention_mask=attention_mask)
        x = self.pooler(x)
        return x

    @classmethod
    def from_pretrained(cls, name: Union[ModelName, str]) -> BGE:
        if not isinstance(name, ModelName):
            name = ModelName(name)

        config = CONFIGS[name]
        bge = BGE.from_config(config)

        hub_dir = torch.hub.get_dir()
        cache_path = os.path.join(hub_dir, "simple-bert-pytorch", config["name"])
        if not os.path.exists(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.hub.download_url_to_file(config["weights_uri"], cache_path)

        state_dict = torch.load(cache_path, weights_only=True)
        state_dict.pop("embeddings.position_ids")
        bge.load_state_dict(state_dict)

        return bge

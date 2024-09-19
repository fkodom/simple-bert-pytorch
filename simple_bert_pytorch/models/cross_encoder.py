from __future__ import annotations

import os
from enum import Enum
from typing import Dict, Optional, Type, TypeVar, Union

import torch
from torch import Tensor, nn

from simple_bert_pytorch.modules import Config, Embeddings, Encoder, Pooler

CrossEncoderType = TypeVar("CrossEncoderType", bound="CrossEncoder")


class ModelName(str, Enum):
    MS_MARCO_MINILM_L_2_V2 = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    MS_MARCO_MINILM_L_4_V2 = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    MS_MARCO_MINILM_L_6_V2 = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MS_MARCO_MINILM_L_12_V2 = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class CrossEncoderConfig(Config):
    name: str
    weights_uri: str

    @classmethod
    def build_for_minilm(
        cls, name: str, weights_uri: str, num_layers: int
    ) -> CrossEncoderConfig:
        return cls(
            name=name,
            weights_uri=weights_uri,
            vocab_size=30522,
            num_layers=num_layers,
            dim=384,
            num_heads=12,
            intermediate_size=1536,
            max_length=512,
            pad_token_id=0,
            dropout=0.1,
            attention_dropout=0.1,
            activation="gelu",
            layer_norm_eps=1e-12,
        )


CONFIGS: Dict[ModelName, CrossEncoderConfig] = {
    ModelName.MS_MARCO_MINILM_L_2_V2: CrossEncoderConfig.build_for_minilm(
        name=ModelName.MS_MARCO_MINILM_L_2_V2.value,
        weights_uri="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-2-v2/resolve/main/pytorch_model.bin",
        num_layers=2,
    ),
    ModelName.MS_MARCO_MINILM_L_4_V2: CrossEncoderConfig.build_for_minilm(
        name=ModelName.MS_MARCO_MINILM_L_4_V2.value,
        weights_uri="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-4-v2/resolve/main/pytorch_model.bin",
        num_layers=4,
    ),
    ModelName.MS_MARCO_MINILM_L_6_V2: CrossEncoderConfig.build_for_minilm(
        name=ModelName.MS_MARCO_MINILM_L_6_V2.value,
        weights_uri="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/pytorch_model.bin",
        num_layers=6,
    ),
    ModelName.MS_MARCO_MINILM_L_12_V2: CrossEncoderConfig.build_for_minilm(
        name=ModelName.MS_MARCO_MINILM_L_12_V2.value,
        weights_uri="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2/resolve/main/pytorch_model.bin",
        num_layers=12,
    ),
}


class CrossEncoder(nn.Module):
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
        self.pooler = Pooler(dim=dim)
        self.classifier = nn.Linear(dim, 1)

    @classmethod
    def from_config(cls: Type[CrossEncoderType], config: Config) -> CrossEncoderType:
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

    @classmethod
    def from_pretrained(cls, name: Union[ModelName, str]) -> CrossEncoder:
        if not isinstance(name, ModelName):
            name = ModelName(name)

        config = CONFIGS[name]
        model = CrossEncoder.from_config(config)

        hub_dir = torch.hub.get_dir()
        cache_path = os.path.join(hub_dir, "simple-bert-pytorch", config["name"])
        if not os.path.exists(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.hub.download_url_to_file(config["weights_uri"], cache_path)

        state_dict: dict[str, Tensor] = torch.load(
            cache_path, weights_only=True, map_location="cpu"
        )
        # NOTE: Because of how cross-encoder models were originally trained in the
        # 'sentence_transformers' library, all weights have a 'bert.' prefix which
        # we do not need.   Strip it off before loading weights into our model.
        state_dict = {k.removeprefix("bert."): v for k, v in state_dict.items()}
        # Position IDs are stored in the cross-encoders checkpoint, but they're not
        # registered as parameters in this implementation -- just as buffers, which
        # are deterministically defined.  We can safely remove them here.
        state_dict.pop("embeddings.position_ids")
        model.load_state_dict(state_dict)

        return model

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        embeddings = self.embeddings.forward(input_ids, position_ids, token_type_ids)
        hidden_states = self.encoder.forward(embeddings, attention_mask)
        pooled = self.pooler.forward(hidden_states)
        return self.classifier.forward(pooled).squeeze(-1)

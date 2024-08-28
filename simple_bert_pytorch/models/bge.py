from __future__ import annotations

import os
from enum import Enum
from typing import Dict, Union

import torch

from simple_bert_pytorch.modules import Bert, Config


class ModelName(str, Enum):
    BGE_SMALL_EN_V1_5 = "BAAI/bge-small-en-v1.5"
    BGE_BASE_EN_V1_5 = "BAAI/bge-base-en-v1.5"
    BGE_LARGE_EN_V1_5 = "BAAI/bge-large-en-v1.5"


CONFIGS: Dict[ModelName, Config] = {
    ModelName.BGE_SMALL_EN_V1_5: Config(
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
    ModelName.BGE_BASE_EN_V1_5: Config(
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
    ModelName.BGE_LARGE_EN_V1_5: Config(
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


class BGE(Bert):
    def from_pretrained(name: Union[ModelName, str]) -> BGE:
        config = CONFIGS[name]
        assert config.weights_uri is not None
        minilm = BGE(
            vocab_size=config.vocab_size,
            dim=config.dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            max_length=config.max_length,
            pad_token_id=config.pad_token_id,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
        )

        hub_dir = torch.hub.get_dir()
        cache_path = os.path.join(hub_dir, "simple-bert-pytorch", config.name)
        if not os.path.exists(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.hub.download_url_to_file(config.weights_uri, cache_path)

        state_dict = torch.load(cache_path, weights_only=True)
        state_dict.pop("pooler.dense.weight")
        state_dict.pop("pooler.dense.bias")
        state_dict.pop("embeddings.position_ids")
        minilm.load_state_dict(state_dict)

        return minilm


if __name__ == "__main__":
    import torch
    from transformers import BertModel

    hf_model = BertModel.from_pretrained(ModelName.BGE_SMALL_EN_V1_5.value).eval()
    model = BGE.from_pretrained(ModelName.BGE_SMALL_EN_V1_5).eval()

    hidden_state = torch.randint(0, hf_model.config.vocab_size, (1, 10))
    attention_mask = torch.rand(1, 10).ge(0.5)
    hf_y = hf_model.forward(hidden_state)
    y = model.forward(hidden_state)

    # print(hf_model.config)

    torch.testing.assert_close(hf_y[0], y, rtol=1e-4, atol=1e-4)

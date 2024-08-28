from __future__ import annotations

import os
from enum import Enum
from typing import Dict, Union

import torch

from simple_bert_pytorch.modules import Bert, Config


class ModelName(str, Enum):
    MSMARCO_MINILM_L_6_V3 = "sentence-transformers/msmarco-MiniLM-L-6-v3"
    MSMARCO_MINILM_L_12_V3 = "sentence-transformers/msmarco-MiniLM-L-12-v3"


class MiniLMConfig(Config):
    weights_uri: str


CONFIGS: Dict[ModelName, MiniLMConfig] = {
    ModelName.MSMARCO_MINILM_L_6_V3: MiniLMConfig(
        name=ModelName.MSMARCO_MINILM_L_6_V3.value,
        weights_uri="https://huggingface.co/sentence-transformers/msmarco-MiniLM-L-6-v3/resolve/main/pytorch_model.bin",
        vocab_size=30522,
        num_layers=6,
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
    ModelName.MSMARCO_MINILM_L_12_V3: MiniLMConfig(
        name=ModelName.MSMARCO_MINILM_L_12_V3.value,
        weights_uri="https://huggingface.co/sentence-transformers/msmarco-MiniLM-L-12-v3/resolve/main/pytorch_model.bin",
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
}


class MiniLM(Bert):
    @classmethod
    def from_pretrained(cls, name: Union[ModelName, str]) -> MiniLM:
        if not isinstance(name, ModelName):
            name = ModelName(name)

        config = CONFIGS[name]
        minilm = MiniLM.from_config(config)

        hub_dir = torch.hub.get_dir()
        cache_path = os.path.join(hub_dir, "simple-bert-pytorch", config["name"])
        if not os.path.exists(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.hub.download_url_to_file(config["weights_uri"], cache_path)

        state_dict = torch.load(cache_path, weights_only=True)
        state_dict.pop("pooler.dense.weight")
        state_dict.pop("pooler.dense.bias")
        state_dict.pop("embeddings.position_ids")
        minilm.load_state_dict(state_dict)

        return minilm


if __name__ == "__main__":
    import torch
    from transformers import BertModel

    hf_model = BertModel.from_pretrained(
        "sentence-transformers/msmarco-MiniLM-L-12-v3"
    ).eval()
    model = MiniLM.from_pretrained(
        "sentence-transformers/msmarco-MiniLM-L-12-v3"
    ).eval()

    state_dict = hf_model.state_dict()
    for key in list(state_dict.keys()):
        if key.startswith("pooler"):
            state_dict.pop(key)
    model.load_state_dict(state_dict)

    hidden_state = torch.randint(0, hf_model.config.vocab_size, (1, 10))
    attention_mask = torch.rand(1, 10).ge(0.5)
    hf_y = hf_model.forward(hidden_state)
    y = model.forward(hidden_state)

    # print(hf_model.config)

    torch.testing.assert_close(hf_y[0], y, rtol=1e-4, atol=1e-4)

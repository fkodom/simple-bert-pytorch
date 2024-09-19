from __future__ import annotations

import os
from enum import Enum
from typing import Dict, Optional, Type, TypeVar, Union

import torch
from torch import Tensor, nn

from simple_bert_pytorch.modules import Backbone, Config, get_activation_fn

BertType = TypeVar("BertType", bound="Bert")


class ModelName(str, Enum):
    BERT_BASE_UNCASED = "bert-base-uncased"
    BERT_LARGE_UNCASED = "bert-large-uncased"
    BERT_BASE_CASED = "bert-base-cased"
    BERT_LARGE_CASED = "bert-large-cased"


class BertConfig(Config):
    name: str
    weights_uri: str


CONFIGS: Dict[ModelName, BertConfig] = {
    ModelName.BERT_BASE_UNCASED: BertConfig(
        name=str(ModelName.BERT_BASE_UNCASED),
        weights_uri="https://huggingface.co/google-bert/bert-base-uncased/resolve/main/pytorch_model.bin",
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
    ModelName.BERT_LARGE_UNCASED: BertConfig(
        name=str(ModelName.BERT_LARGE_UNCASED),
        weights_uri="https://huggingface.co/google-bert/bert-large-uncased/resolve/main/pytorch_model.bin",
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
    ModelName.BERT_BASE_CASED: BertConfig(
        name=str(ModelName.BERT_BASE_CASED),
        weights_uri="https://huggingface.co/google-bert/bert-base-cased/resolve/main/pytorch_model.bin",
        vocab_size=28996,
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
    ModelName.BERT_LARGE_CASED: BertConfig(
        name=str(ModelName.BERT_LARGE_CASED),
        weights_uri="https://huggingface.co/google-bert/bert-large-cased/resolve/main/pytorch_model.bin",
        vocab_size=28996,
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


class BertPredictionHeadTransform(nn.Module):
    def __init__(
        self, dim: int, activation: str = "gelu", layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        if isinstance(activation, str):
            self.transform_act_fn = get_activation_fn(activation)
        else:
            self.transform_act_fn = activation
        self.LayerNorm = nn.LayerNorm(dim, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.transform = BertPredictionHeadTransform(
            dim=dim, activation=activation, layer_norm_eps=layer_norm_eps
        )
        self.decoder = nn.Linear(dim, vocab_size)

    def _tie_weights(self) -> None:
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.predictions = BertLMPredictionHead(
            vocab_size=vocab_size,
            dim=dim,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class Bert(nn.Module):
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
        self.bert = Backbone(
            vocab_size,
            num_layers,
            dim,
            num_heads,
            intermediate_size,
            max_length,
            pad_token_id,
            dropout,
            attention_dropout,
            activation,
            layer_norm_eps,
        )
        self.cls = BertOnlyMLMHead(
            vocab_size=vocab_size,
            dim=dim,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )

    @classmethod
    def from_config(cls: Type[BertType], config: Config) -> BertType:
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
    def from_pretrained(cls, name: Union[ModelName, str]) -> Bert:
        config = CONFIGS[name]
        bert = Bert.from_config(config)

        hub_dir = torch.hub.get_dir()
        cache_path = os.path.join(hub_dir, "simple-bert-pytorch", config["name"])
        if not os.path.exists(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.hub.download_url_to_file(config["weights_uri"], cache_path)

        # Load the pretrained state dict.
        # NOTE: We have to rename some keys to match our implementation.
        state_dict = torch.load(cache_path, weights_only=True)
        # The original BERT models use gamma/beta instead of weight/bias in the
        # LayerNorm layers.  We need to rename those variables here.
        state_dict = {
            k.replace("gamma", "weight").replace("beta", "bias"): v
            for k, v in state_dict.items()
        }
        # For some reason, 'bias' was not contained in the 'decoder' (Linear) layer.
        state_dict["cls.predictions.decoder.bias"] = state_dict.pop(
            "cls.predictions.bias"
        )
        # These weights are not used by the end-to-end model.
        state_dict.pop("bert.pooler.dense.weight")
        state_dict.pop("bert.pooler.dense.bias")
        state_dict.pop("cls.seq_relationship.weight")
        state_dict.pop("cls.seq_relationship.bias")
        bert.load_state_dict(state_dict)

        return bert

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        hidden_states = self.bert.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        return self.cls(hidden_states)


class BertForMaskedLM(Bert):
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        hidden_states = self.bert.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        pooled = self.pooler(hidden_states)
        return self.cls(pooled)


if __name__ == "__main__":
    import torch
    from transformers import BertForMaskedLM

    hf_model = BertForMaskedLM.from_pretrained(ModelName.BERT_BASE_CASED.value).eval()
    model = Bert.from_pretrained(ModelName.BERT_BASE_CASED).eval()

    hidden_state = torch.randint(0, hf_model.config.vocab_size, (1, 10))
    attention_mask = torch.rand(1, 10).ge(0.5)
    hf_y = hf_model.forward(hidden_state, attention_mask)
    y = model.forward(hidden_state, attention_mask)

    torch.testing.assert_close(hf_y.logits, y, rtol=1e-4, atol=1e-4)
    print(y.shape, hf_y[0].shape)

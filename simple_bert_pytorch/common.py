from enum import Enum
from typing import TypedDict


class ModelName(str, Enum):
    # BERT masked language models
    BERT_BASE_UNCASED = "bert-base-uncased"
    BERT_LARGE_UNCASED = "bert-large-uncased"
    BERT_BASE_CASED = "bert-base-cased"
    BERT_LARGE_CASED = "bert-large-cased"

    # BGE embedding models
    BGE_SMALL_EN_V1_5 = "BAAI/bge-small-en-v1.5"
    BGE_BASE_EN_V1_5 = "BAAI/bge-base-en-v1.5"
    BGE_LARGE_EN_V1_5 = "BAAI/bge-large-en-v1.5"

    # Cross-encoder reranker models
    MS_MARCO_MINILM_L_2_V2 = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    MS_MARCO_MINILM_L_4_V2 = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    MS_MARCO_MINILM_L_6_V2 = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MS_MARCO_MINILM_L_12_V2 = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class Config(TypedDict):
    vocab_size: int
    num_layers: int
    dim: int
    num_heads: int
    intermediate_size: int
    max_length: int
    pad_token_id: int
    dropout: float
    attention_dropout: float
    activation: str
    layer_norm_eps: float

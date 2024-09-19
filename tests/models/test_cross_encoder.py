import pytest
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
)

from simple_bert_pytorch.models.cross_encoder import CrossEncoder, ModelName
from simple_bert_pytorch.tokenizer import Tokenizer


@pytest.mark.parametrize(
    "model_name",
    [
        "cross-encoder/ms-marco-MiniLM-L-2-v2",
        "cross-encoder/ms-marco-MiniLM-L-4-v2",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
    ],
)
def test_cross_encoder(model_name: ModelName):
    texts = (
        "The quick brown fox jumps over the lazy dog.",
        "The five boxing wizards jump quickly.",
        "Pack my box with five dozen liquor jugs.",
        "How razorback-jumping frogs can level six piqued gymnasts!",
    )

    # First, get end-to-end outputs from the HuggingFace model
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    hf_tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_tokenized = hf_tokenizer.__call__(texts, padding=True, return_tensors="pt")
    hf_y = hf_model.forward(
        hf_tokenized["input_ids"], hf_tokenized["attention_mask"]
    ).logits.squeeze()

    # Then, get end-to-end outputs from our model
    model = CrossEncoder.from_pretrained(model_name).eval()
    tokenizer = Tokenizer.from_pretrained(model_name)
    tokenized = tokenizer(texts, padding=True, return_tensors="pt")
    y = model.forward(tokenized["input_ids"], tokenized["attention_mask"])

    # Assert that the two output tensors are close
    torch.testing.assert_close(hf_y, y, rtol=1e-4, atol=1e-4)

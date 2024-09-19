import pytest
import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer

from simple_bert_pytorch.models.bge import BGE, ModelName
from simple_bert_pytorch.tokenizer import Tokenizer


@pytest.mark.parametrize(
    "model_name",
    ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"],
)
def test_bge(model_name: ModelName):
    texts = (
        "The quick brown fox jumps over the lazy dog.",
        "The five boxing wizards jump quickly.",
        "Pack my box with five dozen liquor jugs.",
        "How razorback-jumping frogs can level six piqued gymnasts!",
    )

    # First, get end-to-end outputs from the HuggingFace model
    hf_model = AutoModel.from_pretrained(model_name).eval()
    hf_tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_tokenized = hf_tokenizer.__call__(texts, padding=True, return_tensors="pt")
    hf_y = hf_model.forward(
        hf_tokenized["input_ids"], hf_tokenized["attention_mask"]
    ).pooler_output

    # Then, get end-to-end outputs from our model
    model = BGE.from_pretrained(model_name).eval()
    tokenizer = Tokenizer.from_pretrained(model_name)
    tokenized = tokenizer(texts, padding=True, return_tensors="pt")
    y = model.forward(tokenized["input_ids"], tokenized["attention_mask"])

    # Assert that the two output tensors are close
    torch.testing.assert_close(hf_y, y, rtol=1e-4, atol=1e-4)

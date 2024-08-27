from typing import Optional

import pytest
from transformers import BertTokenizer

from simple_bert_pytorch.tokenizer import BasicTokenizer, Tokenizer, WordpieceTokenizer


@pytest.fixture(
    scope="module",
    params=[
        "bert-base-uncased",
        "BAAI/bge-base-en-v1.5",
        "sentence-transformers/msmarco-MiniLM-L-6-v3",
    ],
)
def hf_tokenizer(request: pytest.FixtureRequest) -> BertTokenizer:
    return BertTokenizer.from_pretrained(request.param)


@pytest.fixture(scope="module")
def text(hf_tokenizer: BertTokenizer) -> str:
    return f"""{hf_tokenizer.bos_token}\x01Lorem ipsum dolor sit amet,\t consectetur
adipiscing elit! Sed do eiusmod tempor {hf_tokenizer.cls_token} ut labore et dolore
magna aliqua. {hf_tokenizer.sep_token} \n\rUt enim ad minim veniam, quis nostrud
exercitation ull\x00amco laboris nisi ut aliquip ex ea commodo consequat.
{hf_tokenizer.eos_token}"""


def test_basic_tokenizer(hf_tokenizer: BertTokenizer, text: str):
    basic_tokenizer = BasicTokenizer(
        lower_case=hf_tokenizer.basic_tokenizer.do_lower_case,
        split_on_punc=hf_tokenizer.basic_tokenizer.do_split_on_punc,
    )

    hf_tokens = hf_tokenizer.basic_tokenizer.tokenize(text)
    tokens = basic_tokenizer.tokenize(text)
    assert len(hf_tokens) == len(tokens)
    for t1, t2 in zip(hf_tokens, tokens):
        assert t1 == t2


def test_wordpiece_tokenizer(hf_tokenizer: BertTokenizer, text: str):
    wordpiece_tokenizer = WordpieceTokenizer(
        vocab=hf_tokenizer.wordpiece_tokenizer.vocab,
        unk_token=hf_tokenizer.unk_token,
        max_token_size=hf_tokenizer.wordpiece_tokenizer.max_input_chars_per_word,
    )

    hf_tokens = hf_tokenizer.wordpiece_tokenizer.tokenize(text)
    tokens = wordpiece_tokenizer.tokenize(text)
    assert len(hf_tokens) == len(tokens)
    for t1, t2 in zip(hf_tokens, tokens):
        assert t1 == t2


@pytest.mark.parametrize(
    "padding, max_length, pad_to_multiple_of, skip_special_tokens",
    [
        (True, 512, 8, False),
        (False, 512, 8, False),
        (True, None, 8, True),
        (True, 256, None, True),
    ],
)
def test_tokenizer(
    hf_tokenizer: BertTokenizer,
    text: str,
    padding: bool,
    max_length: Optional[int],
    pad_to_multiple_of: Optional[int],
    skip_special_tokens: bool,
):
    tokenizer = Tokenizer(
        vocab=hf_tokenizer.vocab, do_lower_case=hf_tokenizer.do_lower_case
    )
    hf_tokenized = hf_tokenizer(
        [text],
        padding=padding,
        truncation=(max_length is not None),
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    tokenized = tokenizer(
        [text],
        padding=padding,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    for key in ["input_ids", "attention_mask"]:
        assert len(hf_tokenized[key]) == len(tokenized[key])
        for t1, t2 in zip(hf_tokenized[key][0], tokenized[key][0]):
            assert t1 == t2

    hf_decoded = hf_tokenizer.decode(
        hf_tokenized["input_ids"][0], skip_special_tokens=skip_special_tokens
    )
    decoded = tokenizer.decode(
        tokenized["input_ids"][0], skip_special_tokens=skip_special_tokens
    )
    assert hf_decoded.lower() == decoded.lower()

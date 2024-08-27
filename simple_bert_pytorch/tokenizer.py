import copy
import re
import unicodedata
from collections import OrderedDict
from functools import lru_cache
from math import ceil
from typing import Literal, TypedDict, overload

import torch

# Precompute common characters that will be removed from text
INVALID_CHARS = {chr(0), chr(0xFFFD)}


class BasicTokenizer:
    def __init__(
        self,
        lower_case: bool = True,
        split_on_punc: bool = True,
        special_tokens: set[str] | None = None,
    ):
        self.do_lower_case = lower_case
        self.do_split_on_punc = split_on_punc
        self.special_tokens = special_tokens or set()
        if self.do_lower_case:
            self.special_tokens = {token.lower() for token in self.special_tokens}

    def tokenize(self, text: str) -> list[str]:
        text = clean_text(text)
        if self.do_lower_case:
            text = text.lower()

        special_tokens = "|".join(re.escape(token) for token in self.special_tokens)
        if self.do_split_on_punc:
            # Separate punctuation into their own tokens, then split on any
            # remaining whitespace.  Regex makes this very efficient.
            tokens: list[str] = re.findall(rf"{special_tokens}|[\w']+|[^\w\s]", text)
        else:
            # Only split on special characters and whitespace.
            tokens: list[str] = re.findall(rf"{special_tokens}|[\w']+|\s", text)
            # tokens = text.strip().split()

        # Remove any empty tokens before returning.
        return [token for token in tokens if token]


@lru_cache(maxsize=8192)
def _is_control(char: str) -> bool:
    # Checks whether `char` is a control character.  In practice, caching speeds
    # this up a lot, because most characters are repeated throughout the text.
    return unicodedata.category(char)[0] == "C"


def clean_text(text: str) -> str:
    # Replace tabs, newlines, and return characters with a single space
    text = text.translate(str.maketrans("\t\n\r", "   "))
    # Remove invalid characters and control characters
    control_characters = {c for c in set(text) if _is_control(c)}
    remove_characters = control_characters.union(INVALID_CHARS)
    text = text.translate(str.maketrans("", "", "".join(remove_characters)))
    # Normalize unicode characters
    text = unicodedata.normalize("NFC", text)

    return text


class WordpieceTokenizer:
    def __init__(self, vocab: set[str], unk_token: str, max_token_size: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_token_size = max_token_size

    def tokenize(self, text: str) -> list[str]:
        return [
            token
            for word in text.strip().split()
            for token in _tokenize_word(
                word,
                vocab=self.vocab,
                unk_token=self.unk_token,
                max_token_size=self.max_token_size,
            )
        ]


def _tokenize_word(
    word: str,
    vocab: set[str],
    unk_token: str,
    max_token_size: int,
) -> list[str]:
    word_length = len(word)
    if word_length > max_token_size:
        return [unk_token]

    tokens: list[str] = []
    start = 0

    while start < word_length:
        end = word_length
        while end > start:
            token = word[start:end]
            if start > 0:
                token = f"##{token}"

            if token in vocab:
                tokens.append(token)
                start = end
                break

            end -= 1
        else:
            # If no valid token is found, return unk_token
            return [unk_token]

    return tokens


ListReturnType = TypedDict(
    "ListReturnType", {"input_ids": list[list[int]], "attention_mask": list[list[int]]}
)
TensorReturnType = TypedDict(
    "TensorReturnType",
    {"input_ids": torch.LongTensor, "attention_mask": torch.BoolTensor},
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[str, int],
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    ):
        if do_lower_case:
            pad_token = pad_token.lower()
            cls_token = cls_token.lower()
            sep_token = sep_token.lower()
            mask_token = mask_token.lower()
            vocab = {k.lower(): v for k, v in vocab.items()}

        # Add special tokens to the vocab before saving as a class attribute.
        token_to_id = copy.deepcopy(vocab)
        special_tokens = (unk_token, sep_token, pad_token, cls_token, mask_token)
        for token in special_tokens:
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)

        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.token_to_id = token_to_id
        self.id_to_token = OrderedDict(
            [(id_, token) for token, id_ in self.token_to_id.items()]
        )
        self.basic_tokenizer = BasicTokenizer(
            lower_case=do_lower_case, special_tokens=special_tokens
        )
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=set(self.token_to_id), unk_token=str(unk_token)
        )

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def tokenize(self, text: str) -> list[str]:
        return [
            token
            for word in self.basic_tokenizer.tokenize(text)
            for token in self.wordpiece_tokenizer.tokenize(word)
        ]

    @overload
    def __call__(
        self,
        texts: list[str],
        padding: bool = False,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_tensors: Literal[False] = False,
    ) -> ListReturnType:
        pass

    @overload
    def __call__(
        self,
        texts: list[str],
        padding: bool = False,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_tensors: Literal[True] = True,
    ) -> TensorReturnType:
        pass

    def __call__(
        self,
        texts: list[str],
        padding: bool = False,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_tensors: Literal[True, False] = False,
    ):
        if (
            (max_length is not None)
            and (pad_to_multiple_of is not None)
            and (max_length % pad_to_multiple_of != 0)
        ):
            raise ValueError(
                "If both `max_length` and `pad_to_multiple_of` are provided, "
                "`max_length` must be a multiple of `pad_to_multiple_of`."
            )

        if self.do_lower_case:
            texts = [t.lower() for t in texts]

        tokenized = [[self.cls_token] + self.tokenize(text) for text in texts]
        if max_length is not None:
            tokenized = [tokens[: max_length - 1] for tokens in tokenized]
        tokenized = [tokens + [self.sep_token] for tokens in tokenized]

        if padding:
            pad_token = self.pad_token
            pad_length = max(len(t) for t in tokenized)
            if max_length is not None:
                pad_length = min(max_length, pad_length)
            if pad_to_multiple_of is not None:
                pad_length = ceil(pad_length / pad_to_multiple_of) * pad_to_multiple_of

            for tokens in tokenized:
                tokens += [pad_token] * (pad_length - len(tokens))

        input_ids = [[self.token_to_id[t] for t in tokens] for tokens in tokenized]
        attention_masks = [
            [(0 if t == self.pad_token else 1) for t in tokens] for tokens in tokenized
        ]

        if return_tensors:
            return {
                "input_ids": torch.as_tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.as_tensor(attention_masks, dtype=torch.bool),
            }
        else:
            return {"input_ids": input_ids, "attention_mask": attention_masks}

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        tokens = [self.id_to_token[id_] for id_ in token_ids]
        if skip_special_tokens:
            special_tokens = {
                self.cls_token,
                self.sep_token,
                self.pad_token,
                self.mask_token,
            }
            tokens = [t for t in tokens if t not in special_tokens]

        text = " ".join(tokens).replace(" ##", "").strip()
        if clean_up_tokenization_spaces:
            # NOTE: This could likely be more efficient with regex, but I don't
            # think it matters.  It takes a negligible amount of time, compared to
            # other parts of the tokenization process.
            text = (
                text.replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ' ", "'")
                .replace(" n't", "n't")
                .replace(" 'm", "'m")
                .replace(" 's", "'s")
                .replace(" 've", "'ve")
                .replace(" 're", "'re")
            )

        return text


if __name__ == "__main__":
    from transformers import BertTokenizer
    # from transformers.models.bert.tokenization_bert import load_vocab

    hf_tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text = f"""{hf_tokenizer.bos_token}\x01Lorem ipsum dolor sit amet,\t consectetur
    adipiscing elit! Sed do eiusmod tempor {hf_tokenizer.cls_token} ut labore et dolore
    magna aliqua. {hf_tokenizer.sep_token} \n\rUt enim ad minim veniam, quis nostrud
    exercitation ull\x00amco laboris nisi ut aliquip ex ea commodo consequat.
    {hf_tokenizer.eos_token}"""
    text = "\n".join([text] * 10)

    tokenizer = Tokenizer(
        vocab=hf_tokenizer.vocab,
        do_lower_case=hf_tokenizer.do_lower_case,
        unk_token=hf_tokenizer.unk_token,
        sep_token=hf_tokenizer.sep_token,
        pad_token=hf_tokenizer.pad_token,
        cls_token=hf_tokenizer.cls_token,
        mask_token=hf_tokenizer.mask_token,
    )

    # ENCODE A BATCH OF TEXTS
    texts = [text] * 10
    hf_encoded = hf_tokenizer(
        texts, padding=True, truncation=True, max_length=512, pad_to_multiple_of=8
    )

    encoded = tokenizer(texts, padding=True, max_length=512, pad_to_multiple_of=8)
    hf_dedocded = hf_tokenizer.decode(hf_encoded["input_ids"][0]).lower()
    decoded = tokenizer.decode(encoded["input_ids"][0])
    assert hf_dedocded == decoded

    # print(hf_encoded)
    for key in ["input_ids", "attention_mask"]:
        for hf, enc in zip(hf_encoded[key], encoded[key]):
            assert len(hf) == len(enc), f"{len(hf)} != {len(enc)}"
            for x, y in zip(hf, enc):
                assert x == y
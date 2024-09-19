# simple-bert-pytorch

A very simple BERT implementation in PyTorch, which only depends on PyTorch itself.  Includes pre-trained models, tokenizers, and usage examples.

Coincidentally, the `Tokenizer` implementation in this project is 6-7x faster than the one in the `transformers` library!  There is a lot of unneeded complexity/overhead in `transformers`, which is why I created this project in the first place.

## Usage

TODO

## Install

From PyPI:
```bash
pip install simple-bert-pytorch
```

From source:
```bash
pip install "simple-bert-pytorch @ git+ssh://git@github.com/fkodom/simple-bert-pytorch.git"
```

For contributors:
```bash
# Install all dev dependencies (tests etc.)
pip install "simple-bert-pytorch[test] @ git+ssh://git@github.com/fkodom/simple-bert-pytorch.git"

# Setup pre-commit hooks
pre-commit install
```

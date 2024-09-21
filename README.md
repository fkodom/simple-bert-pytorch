# simple-bert-pytorch

A very simple BERT implementation in PyTorch, which only depends on PyTorch itself.  Includes pre-trained models, tokenizers, and usage examples.

Coincidentally, the `Tokenizer` implementation in this project is 6-7x faster than the one in the `transformers` library!  There is a lot of unneeded complexity/overhead in `transformers`, which is why I created this project in the first place.

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

## Usage

```python
from simple_bert_pytorch.tokenizer import Tokenizer
from simple_bert_pytorch.models.bert import Bert

# You can also load a Tokenizer by passing the `lower_case` argument.  Essentially
# all BERT models use one of 2 vocabularies (cased or uncased).  If you know that
# your model is cased/uncased, this is equivalent to laoding by name.
#     tokenizer = Tokenizer.from_pretrained(lower_case=True)
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

# Similar to `transformers`, pretrained models are loaded using the `from_pretrained`
# method.  But you can also instantiate models directly!  We keep it simple by using
# keyword arguments, rather than config objects, so you can easily see what you're
# passing in.
#     model = Bert(
#         vocab_size=tokenizer.vocab_size,
#         num_layers=6,
#         dim=512,
#         num_heads=8,
#         intermediate_size=2048,
#     )
model = Bert.from_pretrained("bert-base-uncased")

texts = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
]
# The tokenizer accepts a sequence of text strings.  If you don't provide any other
# arguments, the returned "input_ids" and "attention_mask" are lists of lists.  You
# can also truncate, pad, and convert them to tensors in one step:
#     tokenized = tokenizer(texts, max_length=128, padding=True, return_tensors=True)
tokenized = tokenizer(texts)
print(tokenized)
# {
#     'input_ids': [[101, 9850, 24727, ...], [101, 1736, 2079, ...]]
#     'attention_mask': [[1, 1, 1, ...], [1, 1, 1, ...]]
# }

# Check that decoding the tokenized inputs works as expected!  This model is uncased,
# so the texts will be lowercased.
decoded = [
    tokenizer.decode(input_ids, skip_special_tokens=True)
    for input_ids in tokenized["input_ids"]
]
print(decoded)
# [
#     "lorem ipsum dolor sit amet, consectetur adipiscing elit.",
#     "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
# ]

# The model arguments have the same names as the tokenizer keys!  You can implicitly
# pass keyword arguments using the `**` operator, but I prefer to be explicit.
# NOTE: For this example, let use `return_tensors=True` so we don't have to worry
# about padding the input tensors.
tokenized = tokenizer(texts, padding=True, return_tensors=True)
# Pretrained BERT is a masked language model.  It returns logits for each input token.
# NOTE: This is a significant difference from the `transformers` library!  Rather than
# returning a dict or tuple of outputs, the model returns a single Tensor.  This library
# is simple enough that, if you need different outputs, you can easily modify the model
# code to make that happen.
logits = model(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"])
print(logits)
# tensor([[[ -7.5046,  -7.4059,  -7.4317,  ...,  -6.8106,  -6.6921,  -4.7529],
#          ...,
#          [-13.6784, -13.2786, -13.8128,  ..., -11.6681, -12.6777,  -7.2111]]],
#        grad_fn=<ViewBackward0>)
print(logits.shape)
# torch.Size([2, 25, 30522])
```

Each pretrained model may return slightly different outputs, depending on the model
architecture:
| Model Type | Outputs | Example Model |
|------------|---------|---------------|
| Masked Language Model | logits for each input token | `Bert` |
| Embedding Model | single embedding vector for each input sequence | `BGE` |
| Sequence Classification Model<br>(AKA "rerankers") | classification score for each input sequence | `CrossEncoder` |

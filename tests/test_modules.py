import pytest
import torch
from transformers.models.bert.modeling_bert import BertModel as HfBertModel

from simple_bert_pytorch.modules import (
    Attention,
    Bert,
    Embeddings,
    Encoder,
    Intermediate,
    Layer,
    Output,
    Pooler,
    SelfAttention,
    SelfAttentionOutput,
)

# Make CUDA ops fully deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@pytest.fixture(
    scope="module",
    params=[
        "bert-base-uncased",
        "BAAI/bge-base-en-v1.5",
        "sentence-transformers/msmarco-MiniLM-L-6-v3",
    ],
)
def hf_bert_model(request: pytest.FixtureRequest):
    model_name = request.param
    hf_model: HfBertModel = HfBertModel.from_pretrained(model_name)
    return hf_model.eval()


def test_embeddings(hf_bert_model: HfBertModel):
    config = hf_bert_model.config
    hf_embeddings = hf_bert_model.embeddings
    embeddings = Embeddings(
        vocab_size=config.vocab_size,
        dim=config.hidden_size,
        max_length=config.max_position_embeddings,
        pad_token_id=config.pad_token_id,
        layer_norm_eps=config.layer_norm_eps,
        dropout=config.hidden_dropout_prob,
    ).eval()

    embeddings.load_state_dict(hf_embeddings.state_dict(), strict=True)
    token_ids = torch.randint(0, config.vocab_size, (1, 10))
    hf_y = hf_embeddings.forward(token_ids)
    y = embeddings.forward(token_ids)

    torch.testing.assert_close(hf_y, y, rtol=1e-4, atol=1e-4)


def test_self_attention(hf_bert_model: HfBertModel):
    config = hf_bert_model.config
    hf_attention = hf_bert_model.encoder.layer[0].attention.self
    attention = SelfAttention(
        dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        dropout=config.attention_probs_dropout_prob,
    ).eval()

    attention.load_state_dict(hf_attention.state_dict(), strict=True)
    hidden_states = torch.randn(1, 10, config.hidden_size)
    attention_mask = torch.rand(1, 10, 10).ge(0.5)
    # NOTE: We only keep the first output (hidden state), but HuggingFace returns
    # a bunch of other tensors.  We can ignore them for this test.
    hf_y, *_ = hf_attention.forward(hidden_states, attention_mask)
    y = attention.forward(hidden_states, attention_mask)

    torch.testing.assert_close(hf_y, y, rtol=1e-4, atol=1e-4)


def test_self_attention_output(hf_bert_model: HfBertModel):
    config = hf_bert_model.config
    hf_output = hf_bert_model.encoder.layer[0].attention.output
    output = SelfAttentionOutput(
        dim=config.hidden_size,
        layer_norm_eps=config.layer_norm_eps,
        dropout=config.hidden_dropout_prob,
    ).eval()

    output.load_state_dict(hf_output.state_dict(), strict=True)
    hidden_states = torch.randn(1, 10, config.hidden_size)
    attention_output = torch.randn(1, 10, config.hidden_size)
    hf_y = hf_output.forward(hidden_states, attention_output)
    y = output.forward(hidden_states, attention_output)

    torch.testing.assert_close(hf_y, y, rtol=1e-4, atol=1e-4)


def test_attention(hf_bert_model: HfBertModel):
    config = hf_bert_model.config
    hf_attention = hf_bert_model.encoder.layer[0].attention
    attention = Attention(
        dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        dropout=config.hidden_dropout_prob,
        layer_norm_eps=config.layer_norm_eps,
    ).eval()

    attention.load_state_dict(hf_attention.state_dict(), strict=True)
    hidden_states = torch.randn(1, 10, config.hidden_size)
    attention_mask = torch.rand(1, 10, 10).ge(0.5)
    # NOTE: We only keep the first output (hidden state), but HuggingFace returns
    # a bunch of other tensors.  We can ignore them for this test.
    hf_y, *_ = hf_attention.forward(hidden_states, attention_mask)
    y = attention.forward(hidden_states, attention_mask)

    torch.testing.assert_close(hf_y, y, rtol=1e-4, atol=1e-4)


def test_intermediate(hf_bert_model: HfBertModel):
    config = hf_bert_model.config
    hf_intermediate = hf_bert_model.encoder.layer[0].intermediate
    intermediate = Intermediate(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        activation=config.hidden_act,
    ).eval()

    intermediate.load_state_dict(hf_intermediate.state_dict(), strict=True)
    hidden_states = torch.randn(1, 10, config.hidden_size)
    hf_y = hf_intermediate.forward(hidden_states)
    y = intermediate.forward(hidden_states)

    torch.testing.assert_close(hf_y, y, rtol=1e-4, atol=1e-4)


def test_output(hf_bert_model: HfBertModel):
    config = hf_bert_model.config
    hf_output = hf_bert_model.encoder.layer[0].output
    output = Output(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        dropout=config.hidden_dropout_prob,
        layer_norm_eps=config.layer_norm_eps,
    ).eval()

    output.load_state_dict(hf_output.state_dict(), strict=True)
    intermediate_outputs = torch.randn(1, 10, config.intermediate_size)
    attention_outputs = torch.randn(1, 10, config.hidden_size)
    hf_y = hf_output.forward(intermediate_outputs, attention_outputs)
    y = output.forward(intermediate_outputs, attention_outputs)

    torch.testing.assert_close(hf_y, y, rtol=1e-4, atol=1e-4)


def test_layer(hf_bert_model: HfBertModel):
    config = hf_bert_model.config
    hf_layer = hf_bert_model.encoder.layer[0]
    layer = Layer(
        dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        activation=config.hidden_act,
        dropout=config.hidden_dropout_prob,
        attention_dropout=config.attention_probs_dropout_prob,
        layer_norm_eps=config.layer_norm_eps,
    ).eval()

    layer.load_state_dict(hf_layer.state_dict(), strict=True)
    hidden_states = torch.randn(1, 10, config.hidden_size)
    attention_mask = torch.rand(1, 10, 10).ge(0.5)
    hf_y, *_ = hf_layer.forward(hidden_states, attention_mask)
    y = layer.forward(hidden_states, attention_mask)

    torch.testing.assert_close(hf_y, y, rtol=1e-4, atol=1e-4)


def test_encoder(hf_bert_model: HfBertModel):
    config = hf_bert_model.config
    hf_encoder = hf_bert_model.encoder
    encoder = Encoder(
        num_layers=config.num_hidden_layers,
        dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        activation=config.hidden_act,
        dropout=config.hidden_dropout_prob,
        attention_dropout=config.attention_probs_dropout_prob,
        layer_norm_eps=config.layer_norm_eps,
    ).eval()

    encoder.load_state_dict(hf_encoder.state_dict(), strict=True)
    hidden_states = torch.randn(1, 10, config.hidden_size)
    attention_mask = torch.rand(1, 10, 10).ge(0.5)
    hf_y = hf_encoder.forward(hidden_states, attention_mask).last_hidden_state
    y = encoder.forward(hidden_states, attention_mask)

    torch.testing.assert_close(hf_y, y, rtol=1e-4, atol=1e-4)


def test_bert(hf_bert_model: HfBertModel):
    config = hf_bert_model.config
    bert = Bert(
        vocab_size=config.vocab_size,
        dim=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        activation=config.hidden_act,
        dropout=config.hidden_dropout_prob,
        attention_dropout=config.attention_probs_dropout_prob,
        layer_norm_eps=config.layer_norm_eps,
        pad_token_id=config.pad_token_id,
        max_length=config.max_position_embeddings,
    ).eval()

    # NOTE: We choose to treat the 'pooler' and base 'Bert' class as separate
    # components.  The pooler is model-specific, and does not need to be conflated
    # with the basic BERT architecture.  We can remove it here, and we'll test that
    # the pooling mechanism works correctly for each sub-model.
    state_dict = hf_bert_model.state_dict()
    for key in list(state_dict.keys()):
        if "pooler" in key:
            del state_dict[key]

    bert.load_state_dict(state_dict, strict=True)
    token_ids = torch.randint(0, config.vocab_size, (1, 10))
    # NOTE: HuggingFace uses a different attention mask format for the top-level
    # model inputs.  Use a 2D matrix with shape (batch_size, sequence_length)
    attention_mask = torch.rand(1, 10).ge(0.5)
    hf_y = hf_bert_model.forward(token_ids, attention_mask).last_hidden_state
    y = bert.forward(token_ids, attention_mask.unsqueeze(0))

    torch.testing.assert_close(hf_y, y, rtol=1e-4, atol=1e-4)


def test_pooler(hf_bert_model: HfBertModel):
    config = hf_bert_model.config
    hf_pooler = hf_bert_model.pooler
    pooler = Pooler(dim=config.hidden_size).eval()

    pooler.load_state_dict(hf_pooler.state_dict(), strict=True)
    hidden_states = torch.randn(1, 10, config.hidden_size)
    hf_y = hf_pooler.forward(hidden_states)
    y = pooler.forward(hidden_states)

    torch.testing.assert_close(hf_y, y, rtol=1e-4, atol=1e-4)

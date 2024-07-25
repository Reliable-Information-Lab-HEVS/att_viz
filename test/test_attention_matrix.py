import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from copy import deepcopy
from ..att_viz.attention_aggregation_method import AttentionAggregationMethod
from ..att_viz.attention_matrix import AttentionMatrix


def get_completion_matrix():
    model_name = "Salesforce/codegen-350M-mono"

    m = AutoModelForCausalLM.from_pretrained(model_name)
    t = AutoTokenizer.from_pretrained(model_name)

    model_input = t.encode(
        "Hello world,",
        return_tensors="pt",
    )

    gen = m.generate(
        model_input,
        max_new_tokens=10,
        min_new_tokens=0,
        do_sample=False,
        output_attentions=True,
        return_dict_in_generate=True,
    )

    attentions = gen["attentions"]

    del m
    del t
    gc.collect()

    return attentions


def test_constructor_on_good_input():
    num_layers = 20
    num_heads = 16

    attn_matrix = get_completion_matrix()

    a = AttentionMatrix(attn_matrix)

    assert a.num_layers == num_layers

    assert a.num_heads == num_heads

    assert a.attention_matrix == attn_matrix

    assert a.is_formatted is False

    assert (
        a.__repr__() == f"AttentionMatrix ({num_heads} head(s), {num_layers} layer(s))"
    )

    assert (
        a.__str__() == f"AttentionMatrix ({num_heads} head(s), {num_layers} layer(s))"
    )


def test_formatting_no_aggregation_no_zeroing_of_first_attn():
    num_layers = 20
    num_heads = 16
    attn_matrix = get_completion_matrix()
    num_response_tokens = len(attn_matrix)
    num_prompt_tokens = len(attn_matrix[0][0][0][0][0])

    a = AttentionMatrix(deepcopy(attn_matrix))

    assert a.num_heads == num_heads
    assert a.num_layers == num_layers
    assert a.is_formatted is False

    a.format(AttentionAggregationMethod.NONE, zero_first_attention=None)

    assert a.is_formatted is True
    assert a.num_heads == num_heads
    assert a.num_layers == num_layers

    for layer in range(a.num_layers):
        for head in range(a.num_heads):
            for token in range(num_response_tokens):
                for prev_token in range(0, num_prompt_tokens + token):
                    ## -1 for the first attn vector which is actually a lower triangular matrix containing attentions from prompt tokens to prev prompt tokens
                    assert (
                        a.attention_matrix[layer][head][token][prev_token]
                        == attn_matrix[token][layer][0][head][-1][prev_token]
                    )


def test_formatting_no_aggregation_yes_zeroing_of_first_attn():
    num_layers = 20
    num_heads = 16
    attn_matrix = get_completion_matrix()
    num_response_tokens = len(attn_matrix)
    num_prompt_tokens = len(attn_matrix[0][0][0][0][0])

    a = AttentionMatrix(deepcopy(attn_matrix))

    assert a.num_heads == num_heads
    assert a.num_layers == num_layers
    assert a.is_formatted is False

    a.format(AttentionAggregationMethod.NONE, zero_first_attention=True)

    assert a.is_formatted is True
    assert a.num_heads == num_heads
    assert a.num_layers == num_layers

    for layer in range(a.num_layers):
        for head in range(a.num_heads):
            for token in range(num_response_tokens):
                for prev_token in range(0, num_prompt_tokens + token):
                    ## -1 for the first attn vector which is actually a lower triangular matrix containing attentions from prompt tokens to prev prompt tokens
                    if prev_token == 0:
                        assert a.attention_matrix[layer][head][token][prev_token] == 0
                    else:
                        assert (
                            a.attention_matrix[layer][head][token][prev_token]
                            == attn_matrix[token][layer][0][head][-1][prev_token]
                        )


def test_formatting_headwise_averaging_no_zeroing_of_first_attn():
    num_layers = 20
    num_heads = 16
    attn_matrix = get_completion_matrix()
    num_response_tokens = len(attn_matrix)
    num_prompt_tokens = len(attn_matrix[0][0][0][0][0])

    a = AttentionMatrix(deepcopy(attn_matrix))

    assert a.num_heads == num_heads
    assert a.num_layers == num_layers
    assert a.is_formatted is False

    a.format(AttentionAggregationMethod.HEADWISE_AVERAGING, zero_first_attention=False)

    assert a.is_formatted is True
    assert a.num_heads == 1
    assert a.num_layers == num_layers

    for layer in range(a.num_layers):
        head = 0  # After formatting, we only have 1 attention head (the head dimension "collapses")
        for token in range(num_response_tokens):
            for prev_token in range(0, num_prompt_tokens + token):
                ## -1 for the first attn vector which is actually a lower triangular matrix containing attentions from prompt tokens to prev prompt tokens

                computed_mean = a.attention_matrix[layer][head][token][prev_token]
                expected_mean = torch.mean(
                    torch.Tensor(
                        [
                            attn_matrix[token][layer][0][i][-1][prev_token]
                            for i in range(num_heads)
                        ]
                    )
                )

                assert torch.allclose(torch.tensor(computed_mean), expected_mean)


def test_formatting_headwise_averaging_yes_zeroing_of_first_attn():
    num_layers = 20
    num_heads = 16
    attn_matrix = get_completion_matrix()
    num_response_tokens = len(attn_matrix)
    num_prompt_tokens = len(attn_matrix[0][0][0][0][0])

    a = AttentionMatrix(deepcopy(attn_matrix))

    assert a.num_heads == num_heads
    assert a.num_layers == num_layers
    assert a.is_formatted is False

    a.format(AttentionAggregationMethod.HEADWISE_AVERAGING, zero_first_attention=True)

    assert a.is_formatted is True
    assert a.num_heads == 1
    assert a.num_layers == num_layers

    # Set attention towards the first token to 0
    for layer in range(num_layers):
        for head in range(num_heads):
            for token in range(num_response_tokens):
                attn_matrix[token][layer][0][head][-1][0] = 0

    for layer in range(a.num_layers):
        head = 0  # After formatting, we only have 1 attention head (the head dimension "collapses")
        for token in range(num_response_tokens):
            for prev_token in range(0, num_prompt_tokens + token):
                ## -1 for the first attn vector which is actually a lower triangular matrix containing attentions from prompt tokens to prev prompt tokens

                computed_mean = a.attention_matrix[layer][head][token][prev_token]
                expected_mean = torch.mean(
                    torch.Tensor(
                        [
                            attn_matrix[token][layer][0][i][-1][prev_token]
                            for i in range(num_heads)
                        ]
                    )
                )

                assert torch.allclose(torch.tensor(computed_mean), expected_mean)

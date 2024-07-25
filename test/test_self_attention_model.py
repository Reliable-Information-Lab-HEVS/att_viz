from ..att_viz.self_attention_model import SelfAttentionModel


def test_constructor_on_good_input():
    model_name = "Salesforce/codegen-350M-mono"
    m = SelfAttentionModel(model_name)

    assert m.model_name_or_directory == model_name

    assert m.model.generation_config.top_p is None

    assert m.__repr__() == f"SelfAttentionModel\nModel name or directory:{model_name}"

    assert m.__str__() == f"SelfAttentionModel\nModel name or directory:{model_name}"


def test_generate_text():
    pass  # TODO

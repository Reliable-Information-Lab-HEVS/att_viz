import numpy as np
from ..att_viz.renderer import RenderConfig, Renderer
from ..att_viz.attention_aggregation_method import AttentionAggregationMethod


def test_default_render_config():
    rc = RenderConfig()

    assert rc.y_margin == 30
    assert rc.line_length == 770
    assert rc.num_chars_block == 11
    assert rc.token_width == 110
    assert rc.min_token_width == 20
    assert rc.token_height == 22.5
    assert rc.x_margin == 20
    assert rc.matrix_width == 115


def test_renderer_constructor_on_good_input():
    rc = RenderConfig()

    for aggr_method in [
        AttentionAggregationMethod.NONE,
        AttentionAggregationMethod.HEADWISE_AVERAGING,
    ]:
        r = Renderer(render_config=rc, aggregation_method=aggr_method)

        assert r.render_config == rc

        assert r.aggr_method is aggr_method

        assert r.__repr__() == f"Renderer\nAggregation Method:\n{aggr_method}"

        assert r.__str__() == f"Renderer\nAggregation Method:\n{aggr_method}"


def test_default_renderer_has_none_as_aggregation_method():
    r = Renderer(render_config=RenderConfig())

    assert r.aggr_method is AttentionAggregationMethod.NONE

    assert (
        r.__repr__()
        == f"Renderer\nAggregation Method:\n{AttentionAggregationMethod.NONE}"
    )

    assert (
        r.__str__()
        == f"Renderer\nAggregation Method:\n{AttentionAggregationMethod.NONE}"
    )


def test_compute_token_info():
    r = Renderer(RenderConfig())

    tokens = ["Hello", " World", "\n", "!"]
    info, dy = r.create_token_info(tokens)

    assert dy == 22.5  # token_height

    expected_info = []

    expected_info.append(
        (20, 30, 50, 0)
    )  # "Hello" has x=20, y=30, width= 5/11 * 110 = 55, and does not start with a space
    expected_info.append((70, 30, 60, 1))  # " World"
    expected_info.append((130, 30, 20, 0))  # "\n"
    expected_info.append((20, 52.5, 20, 0))  # "!"

    assert np.allclose(info, expected_info)


def test_format_special_chars():
    tokens = ["Hello", "Ċ", "Ġ▁World</w>", "<0x0A>", "!"]

    result = Renderer(RenderConfig())._format_special_chars(tokens)

    expected = ["Hello", ",", "  World", "\n", "!"]

    assert result == expected


def test_rendering():
    pass  # TODO

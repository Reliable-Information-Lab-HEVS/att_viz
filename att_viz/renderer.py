import math
import os
import uuid
import json
from IPython.display import HTML, Javascript
from .attention_matrix import AttentionMatrix
from .attention_aggregation_method import AttentionAggregationMethod


class RenderConfig:
    """
    Rendering configuration class for specifying JavaScript preferences.
    """

    def __init__(
        self,
        y_margin: float = 30,
        line_length: float = 770,
        num_chars_block: float = 11,
        token_width: float = 110,
        min_token_width: float = 20,
        token_height: float = 22.5,
        x_margin: float = 20,
        matrix_width: float = 115,
    ):
        """
        `RenderConfig` constructor.

        Args:
            y_margin: where the HTML render should begin (on the y-axis)

            line_length: the maximum length of a line of text in the HTML render

            num_chars_block: the number of characters of a `TOKEN_WIDTH`-wide token block. This is used for scaling

            token_width: the width of a token block containing `NUM_CHARS_BLOCK` characters. This is used for scaling

            min_token_width: the minimum width of a token block. This is used to control scaling

            token_height: the height of a token block in the HTML render.

            x_margin: the margin on the x-axis of the HTML render.

            matrix_width: the space between the two attention rendition modes
        """

        self.y_margin = y_margin
        self.x_margin = x_margin
        self.line_length = line_length
        self.num_chars_block = num_chars_block
        self.token_height = token_height
        self.token_width = token_width
        self.min_token_width = min_token_width
        self.matrix_width = matrix_width


class Renderer:
    """
    Renderering class for visualizing self-attention matrices.
    """

    def __init__(
        self,
        render_config: RenderConfig,
        aggregation_method: AttentionAggregationMethod = AttentionAggregationMethod.NONE,
    ):
        """
        `Renderer` constructor.

        Args:
            render_config: the rendering configuration. See `RenderConfig`.

            aggregation_method: the aggregation method of the attention matrix. See `AttentionAggregationMethod`.
        """

        self.render_config = render_config
        self.aggr_method = aggregation_method

    def _create_token_info(
        self,
        tokens: list[str],
        start_x: int,
        start_y: int,
        info: list[tuple[int, int, int, int]] | None,
    ) -> tuple[list[tuple[int, int, int, int]], float]:
        """
        Used for js visualization. Computes the (x, y) coordinates for a list of tokens, given the starting point (start_x, start_y)

        Args:
            tokens: an array of tokens

            start_x: the starting x coordinates

            start_y: the starting y coordinates

            info: the list of (x, y) coordinates this function should append to. Defaults to the empty list.

        Returns:
            a pair (info, dy) where info is the modified array with (x, y) coordinates of the given tokens, and dy is the total height of the computed token sequence.
        """
        if info is None:
            info = []

        dx = 0
        dy = 0

        for t in tokens:

            w = min(
                self.render_config.token_width,
                max(
                    self.render_config.min_token_width,
                    (len(t) / self.render_config.num_chars_block)
                    * self.render_config.token_width,
                ),
            )
            info.append(
                [start_x + dx, start_y + dy, w, 1 if (t.startswith(" ")) else 0]
            )

            dx += w

            if dx > self.render_config.line_length or t == "\n":
                dx = 0
                dy += self.render_config.token_height

        return info, dy

    def create_token_info(
        self, tokens: list[str]
    ) -> tuple[list[tuple[int, int, int, int]], float]:
        """
        Used for js visualization. Computes the (x, y) coordinates for a list of tokens.

        Args:
            tokens: an array of tokens

        Returns:
            a pair (res, dy) where res contains the (x, y) coordinates of the given tokens, and dy is the total height of the computed token sequence.
        """
        all_info = []

        res, dy = self._create_token_info(
            tokens,
            start_x=self.render_config.x_margin,
            start_y=self.render_config.y_margin,
            info=all_info,
        )

        return res, dy

    def _format_special_chars(self, tokens: list[str]) -> list[str]:
        """
        Replaces common LLM special tokens into their human readable form.

        Args:
            tokens: an array of tokens

        Returns:
            an array of formatted tokens
        """
        return [
            t.replace("Ġ", " ")
            .replace("▁", " ")
            .replace("</w>", "")
            .replace("Ċ", ",")
            .replace("<0x0A>", "\n")
            for t in tokens
        ]

    def _populate_html(self, attn_data: dict, vis_id: int) -> HTML:
        """
        Creates the structure of an HTML file for self-attention visualization, and populates it with the given information.

        Args:
            attn_data: attention-related information:
                name

                the self-attention matrix

                tokens

                prompt length in tokens

                token positioning information

                the total width of the visualization

                the number of heads

                the number of layers

            vis_id: the desired root element id of the HTML document

        Returns:
            The resulting `Ipython.display.HTML` object
        """
        # Compose html
        vis_html = f"""
            <title>att_viz</title>
            <div id="{vis_id}" style="font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;">
                <span style="user-select:none">
                    Layer: <select id="layer"></select>
                </span>
                <div id='vis'></div>
            </div>
        """

        params = {
            "attention": attn_data,
            "root_div_id": vis_id,
        }

        html1 = HTML(
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>'
        )

        html2 = HTML(vis_html)

        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        with open(
            os.path.join(__location__, "attention_viz.js"), mode="r", encoding="UTF-8"
        ) as fp:
            vis_js = fp.read().replace("PYTHON_PARAMS", json.dumps(params))
            html3 = Javascript(vis_js)
            script = (
                '\n<script type="text/javascript">\n' + html3.data + "\n</script>\n"
            )

            head_html = HTML(html1.data + html2.data + script)
            return head_html

    def _make_htmls(
        self,
        tokens: list[str],
        prompt_length: int,
        attention_matrix: AttentionMatrix,
        render_in_chunks: bool = True,
    ) -> list[HTML]:
        """
        Makes one or more HTML visualizations.

        Args:
            tokens: the list of tokens of the prompt and model completion

            prompt_length: the length of the prompt in tokens

            attention_matrix: a formatted `AttentionMatrix` (see `AttentionMatrix.format`)

            render_in_chunks: indicates whether to render in chunks or not (default `True`)

        Returns:
            a list of the resulting `IPython.display.HTML` object(s)
        """

        id_base = f"AttViz-{(uuid.uuid4().hex)}"

        token_info, dy = self.create_token_info(tokens)

        htmls = []

        attn_data = {
            "name": "Response -> Prompt",
            "tokens": tokens,
            "prompt_length": prompt_length,
            "pos": token_info,
            "dy_total": dy,
            "head_start_idx": 0,
        }

        ## If the aggregation method is not none, we will not render in chunks, as some dimensions have collapsed.
        render_in_chunks = (
            render_in_chunks and self.aggr_method == AttentionAggregationMethod.NONE
        )

        if render_in_chunks:

            for layer_idx in range(attention_matrix.num_layers):
                layer_attention = [
                    attention_matrix.attention_matrix[layer_idx]
                ]  # shape: 1 x num_heads x num_res_tokens x num_tokens_before

                for chunk_idx in range(math.ceil(attention_matrix.num_heads / 8)):
                    start = chunk_idx * 8
                    end = min(attention_matrix.num_heads, start + 8)
                    chunk_attention = [layer_attention[0][start:end]]

                    attn_data.update(
                        {
                            "attn": chunk_attention,
                            "num_heads": end - start,
                            "num_layers": 1,
                            "head_start_idx": start,
                        }
                    )

                    # Generate unique div id to enable multiple visualizations in one notebook
                    uid_str = f"Layer-{layer_idx}__Chunk-{chunk_idx}"

                    res = self._populate_html(attn_data, vis_id=f"{id_base}__{uid_str}")
                    htmls.append({"html": res, "name": uid_str})

        else:
            attn_data.update(
                {
                    "attn": attention_matrix.attention_matrix,
                    "num_heads": attention_matrix.num_heads,
                    "num_layers": attention_matrix.num_layers,
                }
            )

            res = self._populate_html(attn_data, vis_id=id_base)  # We keep the base id
            htmls.append({"html": res, "name": id_base})

        return htmls

    def render(
        self,
        tokens: list[str],
        prompt_length: int,
        attention_matrix: AttentionMatrix,
        prettify_tokens: bool = True,
        render_in_chunks: bool = True,
    ) -> None:
        """
        Creates and saves one or more interactive HTML visualizations of the given attention matrix.

        Args:
            tokens: the list of tokens of the prompt and model completion

            prompt_length: the length of the prompt in tokens

            attention_matrix: a formatted `AttentionMatrix` (see `AttentionMatrix.format`)

            prettify_tokens: indicates whether to remove special characters in tokens, e.g. Ġ. (default `True`)

            render_in_chunks: indicates whether to render in chunks or not (default `True`)
        """

        if prettify_tokens:
            tokens = self._format_special_chars(tokens)

        htmls = self._make_htmls(
            tokens, prompt_length, attention_matrix, render_in_chunks
        )

        for html in htmls:
            with open(f"{html['name']}.html", mode="w", encoding="UTF-8") as fp:
                fp.write(html["html"].data)

    def __repr__(self):
        """
        Debugging string representation of `Renderer`
        """
        return f"Renderer\nAggregation Method:\n{self.aggr_method}"

    def __str__(self):
        """
        Regular string representation of `Renderer`
        """
        return self.__repr__()

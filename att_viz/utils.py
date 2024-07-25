import pickle
import gc
from .self_attention_model import SelfAttentionModel
from .renderer import RenderConfig, Renderer
from .attention_matrix import AttentionMatrix
from .attention_aggregation_method import AttentionAggregationMethod


class Experiment:
    """
    A simple attention visualization experiment.
    """

    def __init__(self, model: SelfAttentionModel, renderer: Renderer):
        """
        `Experiment` constructor.

        Args:
            model: the desired self-attention model for this experiment

            renderer: the desired rendering object for this experiment
        """
        self.model = model
        self.renderer = renderer

    def basic_experiment(
        self, prompt: str, aggr_method: AttentionAggregationMethod
    ) -> None:
        """
        A simple text generation experiment, in which the resulting prompt-completion
        pair is visualized with self-attention information in HTML format.

        Args:
            prompt: the prompt to use for text generation

            aggr_method: the aggregation method of the attention matrix. See `AttentionAggregationMethod`
        """
        completion_tokens, attention_matrix, prompt_length = self.model.generate_text(
            prompt, prompt_template=None
        )

        attention_matrix.format(aggr_method, zero_first_attention=False)

        self.renderer.render(
            completion_tokens,
            prompt_length,
            attention_matrix,
            prettify_tokens=True,
            render_in_chunks=(aggr_method == AttentionAggregationMethod.NONE),
        )

    def __repr__(self):
        """
        Debugging string representation of `Experiment`
        """
        return f"Experiment\nModel:{self.model.__repr__()}\nRenderer:{self.renderer.__repr__()})"

    def __str__(self):
        """
        Regular string representation of `Experiment`
        """
        return self.__repr__()


def save_completions(
    model_name_or_directory: str, prompts: list[str], save_prefixes: list[str]
) -> None:
    """
    Load a self-attention model and do inference for the given prompts.

    Args:
        model_name_or_directory: the name of the model to load, or alternatively the directory from which to load the model

        prompts: the list of prompts to use for text generation

        save_prefixes: the list of save prefixes to use for storing inference results - should have the same length as `prompts`
    """
    assert len(prompts) == len(save_prefixes)

    model = SelfAttentionModel(model_name_or_directory)

    for prompt, save_prefix in zip(prompts, save_prefixes):
        _ = model.generate_text(
            prompt=prompt,
            max_new_tokens=512,
            save_prefix=save_prefix,
            prompt_template=None,
        )  # TODO make passing generation args easier or optional
        del _
        gc.collect()

    del model
    gc.collect()


def process_saved_completions(
    render_config: RenderConfig,
    aggregation_method: AttentionAggregationMethod,
    save_prefixes: list[str],
) -> None:
    """
    Render inference results obtained using `save_completions`.

    Args:
        render_config: the rendering configuration. See `RenderConfig`.

        aggregation_method: the aggregation method of the attention matrix. See `AttentionAggregationMethod`

        save_prefixes: the list of save prefixes that have been used for storing inference results
    """

    renderer = Renderer(
        render_config=render_config, aggregation_method=aggregation_method
    )

    for save_prefix in save_prefixes:
        with open(
            f"{save_prefix}_completion_tokens.pickle", "rb"
        ) as fp_completion_tokens, open(
            f"{save_prefix}_attention_matrix.pickle", "rb"
        ) as fp_att, open(
            f"{save_prefix}_input_length.pickle", "rb"
        ) as fp_inp_len:

            completion_tokens: list[str] = pickle.loads(fp_completion_tokens.read())
            attention_matrix: AttentionMatrix = pickle.loads(fp_att.read())
            input_length: int = pickle.loads(fp_inp_len.read())

            attention_matrix.format(aggregation_method, True)
            renderer.render(
                completion_tokens,
                input_length,
                attention_matrix,
                prettify_tokens=True,
                render_in_chunks=(
                    aggregation_method == AttentionAggregationMethod.NONE
                ),
            )

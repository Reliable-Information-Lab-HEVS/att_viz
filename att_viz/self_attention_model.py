import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .attention_matrix import AttentionMatrix


class SelfAttentionModel:
    """
    Wrapper for a self-attention model.

    Loads and stores the model and its corresponding tokenizer.
    """

    def __init__(self, model_name_or_directory: str):
        """
        `SelfAttentionModel` constructor. Loads and stores the indicated model and its corresponding tokenizer.

        Args:
            model_name_or_directory: the name of the model to load, or alternatively the directory from which to load the model

        """

        m, t = self.load_model(model_name_or_directory)
        self.model = m
        self.tokenizer = t
        self.model_name_or_directory = model_name_or_directory

    def load_model(
        self, model_name_or_directory: str
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:  # TODO figure out the true types
        """
        Loads and returns a HuggingFace pretrained model and the corresponding tokenizer.

        Args:
            model_name_or_directory: the name of the model to load, or alternatively the directory from which to load the model

        Returns:
            the loaded model and tokenizer
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_directory, device_map="balanced", attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_directory)
        model.generation_config.top_p = None

        return model, tokenizer

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        save_prefix: str | None = None,
        prompt_template: str | None = "user\n{p}<|endoftext|>\nassistant\n",
    ) -> tuple[list[str], AttentionMatrix, int]:
        """
        Generates text and returns the completion, attention matrix, and prompt length (in tokens).

        Args:
            prompt: the prompt to use for text generation

            max_new_tokens: the maximum number of tokens to be generated

            save_prefix: the prefix to use if saving the computation results (default `None`)

            prompt_template: the prompt template to use for text generation (default: `"user\n{p}<|endoftext|>\nassistant\n"`)

        Returns:
            the generated completion (as a string and as a list of tokens), attention matrix, and prompt length (in tokens)
        """
        model_input = self.tokenizer.encode(
            prompt_template.format(p=prompt) if prompt_template is not None else prompt,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            model_input = model_input.to("cuda")

        input_length = model_input.shape[-1]

        gen = self.model.generate(
            model_input,
            max_new_tokens=max_new_tokens,
            min_new_tokens=0,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
        )

        completion = gen["sequences"][0]
        attentions = gen["attentions"]

        attention_matrix = AttentionMatrix(attentions)
        # completion_string = self.tokenizer.decode(completion)
        completion_tokens = self.tokenizer.convert_ids_to_tokens(completion)

        if save_prefix is not None:
            with open(
                f"{save_prefix}_completion_tokens.pickle", "wb"
            ) as fp_completion_tokens, open(
                f"{save_prefix}_attention_matrix.pickle", "wb"
            ) as fp_att, open(
                f"{save_prefix}_input_length.pickle", "wb"
            ) as fp_inp_len:

                fp_completion_tokens.write(
                    pickle.dumps(completion_tokens, pickle.HIGHEST_PROTOCOL)
                )
                fp_att.write(pickle.dumps(attention_matrix, pickle.HIGHEST_PROTOCOL))
                fp_inp_len.write(pickle.dumps(input_length, pickle.HIGHEST_PROTOCOL))

        return completion_tokens, attention_matrix, input_length

    def __repr__(self):
        """
        Debugging string representation of `SelfAttentionModel`
        """
        return f"SelfAttentionModel\nModel name or directory:{self.model_name_or_directory}"

    def __str__(self):
        """
        Regular string representation of `SelfAttentionModel`
        """
        return self.__repr__()

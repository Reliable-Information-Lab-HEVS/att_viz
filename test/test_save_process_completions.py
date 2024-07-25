import pickle
import torch
from ..att_viz.utils import (
    save_completions,
    process_saved_completions,
)
from ..att_viz.attention_aggregation_method import AttentionAggregationMethod
from ..att_viz.renderer import RenderConfig
from ..att_viz.self_attention_model import SelfAttentionModel
from ..att_viz.attention_matrix import AttentionMatrix


class MockFileOpener:
    def __init__(self):
        self.files = {}

    def open_file(self, filename, *args, **kwards):
        if filename not in self.files:
            new_file_pointer = MockFilePointer()
            self.files.update({filename: new_file_pointer})

        return self.files.get(filename)


class MockFilePointer:
    def __init__(self):
        self.content = ""
        self.num_reads = 0
        self.num_writes = 0

    def write(self, content):
        self.num_writes += 1
        print(self.content)
        self.content = content

    def read(self, *args, **kwargs):
        self.num_reads += 1
        # print(str(self.content))
        return self.content

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


class MockModel:
    class _FakeGenConfig:
        def __init__(self):
            self.top_p = None

    def __init__(self, *args, **kwargs):
        self.generation_config = self._FakeGenConfig()
        self.sequences = [torch.rand(1, 2, 3, 4)]
        self.attentions = torch.rand(2, 2, 2, 2, 2, 2)

    def generate(self, *args, **kwargs):
        return {
            "sequences": self.sequences,
            "attentions": self.attentions,
        }

    def from_pretrained(self, *args, **kwargs):
        return self


class MockTokenizer:
    def __init__(self, *args, **kwargs):
        self.encoded = torch.rand(2, 2, 3, 4)

    def encode(self, *args, **kwargs):
        return self.encoded

    def convert_ids_to_tokens(self, *arg, **kwargs):
        return ["Hello", " World"]

    def from_pretrained(self, *args, **kwargs):
        return self


def test_save_completions(mocker):
    model_name = "Salesforce/codegen-350M-mono"

    prompts = ["Hello, World!", "print('Garfield')"]
    save_prefixes = ["example_0", "example_1"]

    mock_opener = MockFileOpener()

    mock_model = MockModel()
    mock_tokenizer = MockTokenizer()

    mocker.patch(
        "transformers.AutoModelForCausalLM.from_pretrained", mock_model.from_pretrained
    )
    mocker.patch(
        "transformers.AutoTokenizer.from_pretrained", mock_tokenizer.from_pretrained
    )
    mocker.patch("builtins.open", mock_opener.open_file)
    save_completions(model_name, prompts, save_prefixes)

    for save_prefix in save_prefixes:
        for suffix in ["completion_tokens", "attention_matrix", "input_length"]:
            assert f"{save_prefix}_{suffix}.pickle" in mock_opener.files

            mock_pointer = mock_opener.files.get(f"{save_prefix}_{suffix}.pickle")

            assert mock_pointer.num_reads == 0
            assert mock_pointer.num_writes == 1

            if suffix == "completion_tokens":
                assert (
                    pickle.loads(mock_pointer.read())
                    == mock_tokenizer.convert_ids_to_tokens()
                )
            elif suffix == "attention_matrix":
                assert pickle.loads(mock_pointer.read()) == AttentionMatrix(
                    mock_model.attentions
                )
            else:
                assert (
                    pickle.loads(mock_pointer.read())
                    == mock_tokenizer.encoded.shape[-1]
                )


def test_process_completions(mocker):
    model_name = "Salesforce/codegen-350M-mono"

    prompts = ["Hello, World!", "print('Garfield')"]
    save_prefixes = ["example_0", "example_1"]

    mock_opener = MockFileOpener()

    mock_model = MockModel()
    mock_tokenizer = MockTokenizer()

    mocker.patch(
        "transformers.AutoModelForCausalLM.from_pretrained", mock_model.from_pretrained
    )
    mocker.patch(
        "transformers.AutoTokenizer.from_pretrained", mock_tokenizer.from_pretrained
    )
    mocker.patch("builtins.open", mock_opener.open_file)
    save_completions(model_name, prompts, save_prefixes)
    process_saved_completions(
        RenderConfig(), AttentionAggregationMethod.NONE, save_prefixes
    )

    for save_prefix in save_prefixes:
        for suffix in ["completion_tokens", "attention_matrix", "input_length"]:
            assert f"{save_prefix}_{suffix}.pickle" in mock_opener.files

            mock_pointer = mock_opener.files.get(f"{save_prefix}_{suffix}.pickle")

            assert mock_pointer.num_reads == 1
            assert mock_pointer.num_writes == 1

            if suffix == "completion_tokens":
                assert (
                    pickle.loads(mock_pointer.read())
                    == mock_tokenizer.convert_ids_to_tokens()
                )
            elif suffix == "attention_matrix":
                assert pickle.loads(mock_pointer.read()) == AttentionMatrix(
                    mock_model.attentions
                )
            else:
                assert (
                    pickle.loads(mock_pointer.read())
                    == mock_tokenizer.encoded.shape[-1]
                )

    ## Assert 2 and only 2 .html file were created
    html_pointers = [
        file_pointer
        for filename, file_pointer in mock_opener.files.items()
        if filename.endswith(".html")
    ]

    assert len(html_pointers) == 2

    for html_pointer in html_pointers:
        assert html_pointer.num_reads == 0

        assert html_pointer.num_writes == 2

        html_content = html_pointer.read()

        # Basic content checks (todo, maybe: check content against an already-rendered html file)
        assert """<title>att_viz</title>""" in html_content
        assert (
            """style="font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;">"""
            in html_content
        )


def test_pickling_works():
    model_name = "Salesforce/codegen-350M-mono"

    m = SelfAttentionModel(model_name)

    tokens, attnM, promptLen = m.generate_text("Hello, world", 10, prompt_template=None)

    assert tokens == pickle.loads(
        pickle.dumps(tokens, protocol=pickle.HIGHEST_PROTOCOL)
    )

    assert attnM == pickle.loads(pickle.dumps(attnM, protocol=pickle.HIGHEST_PROTOCOL))

    assert promptLen == pickle.loads(
        pickle.dumps(promptLen, protocol=pickle.HIGHEST_PROTOCOL)
    )

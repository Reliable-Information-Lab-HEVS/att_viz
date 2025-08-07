# **<p align=center> att_viz </p>**
### <p align=center> [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md) <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a> [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Coverage Status](https://github.com/aindreias/att_viz/blob/main/reports/coverage/coverage_badge.svg)](https://github.com/aindreias/att_viz/blob/main/reports/coverage/html_report/index.html) </p>

#### <p align=center> [Documentation](https://aindreias.github.io/att_viz/att_viz.html) | [Statement of Need](#statement-of-need)  |  [Target Audience](#target-audience)  |  [Features](#features)  |  [Contributing](#contributing) | [Other Packages](#other-packages-for-visualizing-attention)</p>

# <p align=center> ![Example of att\_viz](https://github.com/aindreias/att_viz/blob/main/examples/tomjerry_v2.png "Title") </p>

## Installation Guide

`att_viz` is published on [TestPyPi](https://test.pypi.org/project/att-viz/0.0.6/). Install it using:

`pip install -i https://test.pypi.org/simple/ att-viz==0.0.6`

Documentation is available [here](https://aindreias.github.io/att_viz/att_viz.html)

## Quick usage

Once installed, `att_viz` integrates directly with the Transformers interface. 

```
from att_viz.utils import Experiment
from att_viz.renderer import Renderer, RenderConfig
from att_viz.self_attention_model import SelfAttentionModel
from att_viz.attention_aggregation_method import AttentionAggregationMethod

model_name_or_directory: str = "openlm-research/open_llama_7b"
# For models hosted by HuggingFace, it is org/name, eg ""
prompt: str = "Continue this story: The beautiful grey cat, Tom, is chasing after Jerry the mouse. He is 100%"

# Initialize the model: this loads the corresponding Huggingface model and tokenizer
model = SelfAttentionModel(model_name_or_directory=model_name_or_directory)

# Initialize the renderer with the base configuration and no attention aggregation method
render_config=RenderConfig()
renderer = Renderer(render_config,
                    aggregation_method=AttentionAggregationMethod.NONE)

# Initialize the experiment 
experiment = Experiment(model, renderer)

# Finally, run inference, save the attention matrices and generate html render files in the current working directory
experiment.basic_experiment(prompt=prompt,
                            aggr_method=AttentionAggregationMethod.NONE,
                            prompt_template=None)
```

Beware that the size of generated pages will be linear in the size of the model and quadratic in the size of the text.

For helping with the interpretability we recommend running the post-processing pipeline with a `python post_processing.py <filename> <amplification> <filter>` documentation about the commands is available with a `python post_processing.py --help`.


## Statement of Need

Advances in the language modelling field have led to the development of self-attention-based large language models (LLMs). These models have billions of learnable parameters and need to be trained on astronomic amounts of data. Their impressive performance relies on complex interactions within the architecture, something which is often difficult to isolate and explain.

LLMs generate each new token by taking context information from previous tokens within a context window - in the order of thousands or more. The contributions of past tokens are weighted using self-attention values, one per attention head per layer. Therefore, to identify which tokens are particulary important for the generation of their successors, one can obtain the self-attention matrix from the LLM and visualize it. Visualizing self-attention can also help study the behaviours of individual attention heads, and ultimately may serve as a diagnostic or debugging tool.

Readily-available visualization tools tend to be less user-friendly, may have outdated dependencies, and cannot handle large amounts of attention data that one would expect from regular-sized models. This is because HTML visualizations are used, and they need to contain the full attention matrix, the size of which depends on the number of layers, attention heads, and the number of prompt and completion tokens. att_viz supports larger completion sizes, as well as bigger models, by splitting the visualization into multiple files when needed. Furthermore, it is a fully-documented Python package, designed to be easily extensible in order to fit specific users' needs.

## Target Audience

This package was developed with researchers, industry practitioners and students in mind. It aims to be a trusty and easy-to-use explainability tool for anybody interested in investigating or further understanding the behaviour of self-attention-based large language models.

## Features

First-time users can gain a deeper understanding of the `att_viz` API through our GitHub Pages documentation, or alternatively by running our introductory Jupyter Notebooks.

With `att_viz`, the first step is always loading a self-attention-based model using `SelfAttentionModel`, running inference on it, and then rendering the resulting self-attention matrix using `Renderer`. The `Experiment` class provides a simple example of this process, through its `basic_experiment` method.

The obtained HTML visualization offers two views:

**1. Observed view**: Hover over a token to see how it has influenced the generation of its successor tokens through self-attention

**2. Observer view**: Hover over a token to see how previous tokens have influenced its generation through self-attention

In both views, users can freeze the attention value visualization for a certain token by double clicking on it. The two views can be (un)frozen independently.

`att_viz` also offers the following features:
- Save model completions and the corresponding self-attention matrices for later. This allows users to separate the generation and visualization tasks. For example, one might want to use GPUs for inference but CPUs for processing the results. The corresponding functions are `save_completions` and `process_saved_completions`;
- Aggregate attention through headwise averaging, while the layer dimension is kept (`AttentionAggregationMethod.HEADWISE_AVERAGING`);
- Break up the visualization into multiple HTML files if the model is too large. One file is created per layer and chunk of eight self-attention heads.

## Contributing

Contributions are welcome. Check out our contribution guide [here](https://github.com/aindreias/att_viz/blob/main/CONTRIBUTING.md).

## Other packages for visualizing attention

`att_viz` started as a modification of these packages, in order to support visualizing large self attention matrices:
- [`bertviz`](https://github.com/jessevig/bertviz) by Jesse Vig
- [`attention`](https://github.com/mattneary/attention/) by Matt Neary

We'd like to think `att_viz` is more convenient to use, but nevertheless we encourage interested users to take a look, especially at `bertviz` for visualizing cross attention in short sentences.

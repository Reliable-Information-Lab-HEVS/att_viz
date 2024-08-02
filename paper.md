---
title: 'att_viz: A Python package for self-attention visualization'
tags:
  - Python
  - LLMs
  - large language models
  - self-attention
authors:
  - name: Ana-Maria Indreias
    affiliation: "1, 2"
  - name: Ljiljana Dolamic
    affiliation: "1"
  - name: Andrei Kucharavy
    affiliation: "1, 3"
affiliations:
  - name: CyberDefence Campus, EPFL Innovation Park, Lausanne, Switzerland
    index: 1
  - name: Ecole Polytechnique Federale de Lausanne, Lausanne, Switzerland
    index: 2
  - name: Haute Ecole Specialisee de Suisse Occidentale Valais, Valais, Switzerland
    index: 3
date: 18 August 2024
bibliography: paper.bib
---

## Summary
`att_viz` is a Python package for visualizing the self-attention matrix of large language models (LLMs). It uses [`bertviz`](https://github.com/jessevig/bertviz), a Python package created for cross-attention visualization, as its starting block.

## Statement of Need
The development of self-attention-based large language models (LLMs) has greatly revolutionized the language modelling field. These models have billions of learnable parameters and need to be trained on considerable amounts of data. Their remarkable performance relies on complex interactions within the architecture, which is often challenging to isolate or interpret.

LLMs generate each new token by taking context information from preceding tokens within a context window - in the order of thousands or more. The contributions of past tokens are weighted using self-attention values, one per attention head per layer. Therefore, to identify which tokens significantly impact the generation of their successors, one could investigate the self-attention values. Visualizing the self-attention matrix can also support the study of the behaviours of individual attention heads, and ultimately may serve as a diagnostic or debugging tool.

Readily available visualization tools tend to be less user-friendly, may have outdated dependencies, and cannot handle large amounts of attention data, which one would expect from regular-sized models. This is because HTML visualizations are used, and they need to contain the full attention matrix, the size of which depends on the number of layers, attention heads, and the number of prompt and completion tokens. `att_viz` supports larger completion sizes, as well as bigger models, by splitting the visualization into multiple files when needed. Furthermore, it is a fully-documented Python package, designed to be easily extensible in order to fit specific users' needs.

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
If the model is too large, Break up the visualization into multiple HTML files. One file is created per layer and per chunk of eight self-attention heads.

## Example Application
A sentence and its rephrased counterpart, though syntactically similar, may be interpreted in a completely different manner by (large) language models. For example, the following natural language prompt pair for the code-generation model `CodeLLama-34B Instruct` leads to the model generating always vulnerable, respectively never vulnerable code:

1. *(Always vulnerable)* In Python, create a Flask web application with a single route that renders an HTML template taken from a jinja2 environment. The rendered template includes a customizable greeting message based on the ’name’ parameter provided in the query string.

2. *(Never vulnerable)* Develop a Flask web application in Python, featuring a solitary route that utilizes a Jinja2 environment to render an HTML template. The template, once rendered, incorporates a personalized greeting message determined by the ’name’ parameter supplied in the query string.

After a closer investigation with `att_viz`, we have discovered that the model focuses on different token types for the two prompts. 

## Similar Applications
We identify the following packages for attention visualization:
- [`bertviz`](https://github.com/jessevig/bertviz), developed by Jesse Vig (supports cross-attention)
- [`attention`](https://github.com/mattneary/attention/), developed by Matt Neary (supports self-attention)
  
* `att_viz` started as a modification of `bertviz` for supporting self-attention instead of cross-attention

* `att_viz` implements self-attention visualizations for bigger models by splitting the matrix in chunks. This ensures the resulting HTML files are small and easy to render.

* Like `attention`, `att_viz` visualizes the prompt and completion horizontally and as a whole, which is more natural than the horizontal approach of `bertviz`, especially for longer texts.

* Like `attention`, `att_viz` supports any model with self-attention from HuggingFace (or any local model). All that is required is a way to obtain the self-attention matrix from the model's `generate` method. On the other hand, `bertviz` supports a limited amount of models.

# Introduction

This repo is to support a blog to help anyone estimate the cost while optionally considering latency for large scale
classification, embedding, or vision embedding. It uses [michaelfeil/infinity](https://github.com/michaelfeil/infinity/)
and [Inference Endpoints](https://huggingface.co/inference-endpoints/dedicated). You can swap either rather easily for
your inference server and deployment method of choice.

I considered a variety of things like:

- GPU type
- Infinity Image type
- Many Batch Sizes
- Many VUs amounts
- Multiple Architectures

# Installation

I used ![Python](https://img.shields.io/badge/python-3.12-blue)

1. `git clone https://github.com/datavistics/encoder-analysis.git`
2. `cd encoder-analysis`
3. `pip install -r requirements.txt`
4. `pip install jupyterlab`
5. [Install k6](https://grafana.com/docs/k6/latest/set-up/install-k6/#install-k6) based on your platform
6. `jupyter lab`
7. Run your notebook of choice

# Project Structure

- There are notebooks in the top level for convenience. Its probably cleaner to put them in `./notebooks` but its
  annoying to add it to path, so I opted for user satisfaction rather than aesthetics
    - **\*-optimization.ipynb** - These were used for generating and conducting the experiments
    - **\*-analysis.ipynb** - These show the analysis in a clean notebook-centric way
    - **\*-analysis-gradio.ipynb** - These show the analysis in an interactive gradio-centric way
- `src` I abstracted a fair amount of code here. I tried to keep any important details in the notebooks
- `templates` these are the k6 jinja templates that I use to generate each experiment
- `data`, `generated`, and `results` are used to store non-version-controlled project files

# Results

Do check out these [notebooks](https://nbviewer.org/github/datavistics/encoder-analysis/tree/main/) in nbviewer, as I
took a lot of effort to make sure they are **interactive**. Unfortunately they look better in light mode due to the
tables.
But follow your heart.

- [classification-analysis-gradio.ipynb](https://nbviewer.org/github/datavistics/encoder-analysis/blob/main/classification-analysis-gradio.ipynb)
- [embedding-analysis-gradio.ipynb](https://nbviewer.org/github/datavistics/encoder-analysis/blob/main/embedding-analysis-gradio.ipynb)
- [vision-embedding-analysis-gradio.ipynb](https://nbviewer.org/github/datavistics/encoder-analysis/blob/main/vision-embedding-analysis-gradio.ipynb)

## Classification

For [lxyuan/distilbert-base-multilingual-cased-sentiments-student](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student)
on a dataset like [tyqiangz/multilingual-sentiments](https://huggingface.co/datasets/tyqiangz/multilingual-sentiments)
(using the `text` column) we can do 1 Billion classifications for only `$253.82`.

| GPU           | Image         | Batch Size | VUs     | Min Cost    |
|---------------|---------------|------------|---------|-------------|
| **nvidia-l4** | **`default`** | **64**     | **448** | **$253.82** |

![classification-results.png](media/classification-results.png)
[Interactive Version here](https://nbviewer.org/github/datavistics/encoder-analysis/blob/main/classification-analysis-gradio.ipynb)

## Embeddings

For [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) on a dataset
like [sentence-transformers/trivia-qa-triplet](https://huggingface.co/datasets/sentence-transformers/trivia-qa-triplet)
(using the `positive` column) we can do 1 Billion embeddings for only `$409.44`.

| GPU       | Batch Size | VUs | Min Cost |
|-----------|------------|-----|----------|
| nvidia-l4 | 256        | 32  | $409.44  |

![embedding-results.png](media/embedding-results.png)
[Interactive Version here](https://nbviewer.org/github/datavistics/encoder-analysis/blob/main/embedding-analysis-gradio.ipynb)

## Vision Embeddings

For [vidore/colqwen2-v1.0-merged](https://huggingface.co/vidore/colqwen2-v1.0-merged) on a dataset
like [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset)
(using the `image` column) we can do 1 Billion ColBERT style embeddings (late interaction) for `$44496.51`.

| GPU       | Batch Size | VUs | Min Cost   |
|-----------|------------|-----|------------|
| nvidia-l4 | 4          | 4   | $44496.51  |

![vision-embedding-results.png](media/vision-embedding-results.png)
[Interactive Version here](https://nbviewer.org/github/datavistics/encoder-analysis/blob/main/embedding-analysis-gradio.ipynb)

# How does it work?

Each of the **\*-optimization.ipynb** notebooks facilitates this structure:

```mermaid
flowchart TD;
    subgraph Benchmarking Server
        A[k6 Running Tests]
        D[Instance Config]
    end

    subgraph Inference Endpoint
        C[Container Running Infinity]
        E[Next Inference Endpoint]
    end

    D -->|Defines Test Parameters| A
    D -->|Deploys Inference Endpoint| E
    A -->|Sends Test Data| C
    C -->|Processes and Returns| A
```

# References and Links

- [Inference Endpoints](https://huggingface.co/inference-endpoints/dedicated)
- [michaelfeil/infinity](https://github.com/michaelfeil/infinity/)
- [Infinity Swagger](https://michaelfeil.eu/infinity/0.0.75/swagger_ui/)
- [k6 Docs](https://grafana.com/docs/k6/latest/)

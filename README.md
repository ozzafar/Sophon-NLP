# SOPHON-NLP: Non-Fine-Tunable Learning to Restrain Task Transferability For Pre-trained Language Models

Oz Zafar, Daniela Gottesman

Tel-Aviv University

## Introduction
[SOPHON](https://arxiv.org/abs/2404.12699), a recent approach introduced at IEEE S&P 2024, aims to pre-train models in such a way that they resist fine-tuning for restrained tasks like generating unsafe content or inferring privacy.
However, prior studies have only tested SOPHON against the images domain and against a limited range of domains and fine-tuning methods, raising questions about its true robustness.
In this project, we evaluate SOPHON on classification tasks in the textual domain (NLP).

**This is a forked repository of the [original SOPHON repository](https://github.com/ChiangE/Sophon) that includes support for language models, see our new code in this PR: https://github.com/ChiangE/Sophon/pull/5.**

<img src="https://github.com/Sophon-NonFinetunableLearning/Sophon/blob/main/sophon.png" width="400" align="center"/>


## Preperation

You can build the required environment by running:

```bash
conda env create -f environment.yml
```
Put pretrained models in ``./classification/pretrained`` and ``./generation/pretrained``

Put datasets in ``../datasets``


## Usage

The usage project is located in two places in two places: 	

+ [classification](classification) : reproducing classification-related SOPHON models
+ [GPT-2 evaluation notebook](classification/experiment_sophon_on_gpt2.ipynb) : shows the evaluation of SOPHON on GPT-2, including all the steps of this work


### Classification

Workspace is ``./classification``, thus

```bash
cd classification
```

#### Train Sophoned model

For inverse cross-entropy sophon, run:

```bash
python inverse_loss.py --alpha 3 --beta 5 --dataset IMDB --arch gpt2
```

The output ckpt will be saved to `results/inverse_loss/[args.arch]_[args.dataset]/[current_time]/`

For kl divergence from uniform distribution sophon, run:

```bash
python kl_uniform_loss.py.py --alpha 1 --beta 1 --nl 5 --dataset IMDB --arch gpt2
```

The argument ``args.dataset`` defines the dataset of the restricted task, while the original task dataset is fixed to ['The PILE'](https://pile.eleuther.ai/).

The choices of ``args.arch`` is currently only ``[gpt2]``.

The output ckpt will be saved to `results/kl_loss/[args.arch]_[args.dataset]/[current_time]/`



## GPT-2 evaluation notebook

This notebook shows a full evaluation of SOPHON on the textual domain, specifically on GPT-2 language model in the sentiment analysis task. The notebook includes:

+ Showing that GPT-2 doesn't perform well on sentiment analysis.
+ Showing that fine-tuning a linear layer on top of GPT-2 performs well on sentiment analysis.
+ Examine if a SOPHONed GPT-2 model is restricted to be fine-tuned on sentiment analysis. The SOPHON ckpt is configurable.


# Multi-Variant Text Analysis and Generation (BERT, GPT, and GANs)

This repository contains the implementation and architectural evaluation of three distinct Natural Language Processing (NLP) paradigms: **Autoencoding** (BERT), **Autoregressive** (GPT), and **Adversarial** (Text-GAN). All models are trained and evaluated on the same textual corpus to compare their structural behaviors, limitations, and operational mechanics.

## Dataset Repository Link
The models in this repository are fine-tuned on the standardized movie review sentiment dataset:
* **Dataset Source:** [Hugging Face - Cornell Movie Review Data (Rotten Tomatoes)](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)


## Analytical Discussion

### 1. Tokenization System Differences
* **BERT (`bert-base-uncased`):** Uses **WordPiece** tokenization. It breaks down text into subwords to minimize out-of-vocabulary (`[UNK]`) issues and appends special structural markers like `[CLS]` (for classification representation) and `[SEP]`.
* **GPT (`distilgpt2`):** Uses **Byte-Pair Encoding (BPE)**. It builds a vocabulary from the ground up starting at the byte level, ensuring smooth causal generation by reading text strictly from left-to-right.
* **Text-GAN:** Uses a **Character/Word-Level Integer Mapping** strategy paired with an index-to-word dictionary. This simple structure maps straight to vector configurations, which is necessary to keep memory overhead manageable during custom adversarial matrix execution.

### 2. The Discrete Text Bottleneck in GANs
Generative Adversarial Networks (GANs) natively struggle with discrete sequence structures like text relative to autoregressive networks like GPT:
* **The Math Breakdown:** GANs were originally designed for continuous feature boundaries (such as image pixel values ranging from `0.0` to `1.0`). If a generator makes an image pixel slightly lighter, the discriminator evaluates a continuous change and yields a smooth gradient back to the generator weights.
* **The Text Problem:** Text tokens are discrete vocabulary integers. Picking a word requires a categorical `argmax()` extraction layer. Because `argmax()` works like a non-continuous step function, its mathematical derivative is **zero everywhere**. This completely blocks standard backpropagation from traveling from the Discriminator back to update the Generator. 
* This project bypasses this issue using a **Gumbel-Softmax** trick to relax discrete samples into continuous, differentiable distributions. However, autoregressive models like GPT naturally excel because they completely avoid an adversarial discriminator, computing smooth cross-entropy loss directly across next-token probability profiles.

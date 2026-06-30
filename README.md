# Multi-Variant Text Analysis and Generation (BERT, GPT, and GANs)

This repository contains the implementation and architectural evaluation of three distinct Natural Language Processing (NLP) paradigms: **Autoencoding** (BERT), **Autoregressive** (GPT), and **Adversarial** (Text-GAN). All models are trained and evaluated on the same textual corpus to compare their structural behaviors, limitations, and operational mechanics.

## Dataset Repository Link
The models in this repository are fine-tuned on the standardized movie review sentiment dataset:
* **Dataset Source:** [Hugging Face - Cornell Movie Review Data (Rotten Tomatoes)](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)

## Google Colab Link
* **Colab Link:** [ACT4 - Multi-Variant Text Analysis and Generation (BERT, GPT, and GANs)](https://colab.research.google.com/drive/1S_78vuDZqzfeshwzXB17CmznyDkv1jYZ?usp=sharing)

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

## Structured Data Processing, Model Pipelines, Training Histories, and Test Inferences.
### 1. Structured Data Processing
* **Location:** `Part 1: Environment Setup & Dataset Initialization` and individual model tokenization pipelines.
* **Mechanism:** Text data is dynamically pulled from the canonical Hugging Face repository and split cleanly into training and testing subsets. The data is structurally processed using dedicated mapping functions (`tokenize_bert`, `tokenize_gpt`, and `encode_text`) to transform raw string reviews into padded, truncated tensor matrices suitable for deep learning inputs.

### 2. Model Pipelines
* **Location:** `Part 2`, `Part 3`, and `Part 4`.
* **Mechanism:** The repository establishes three distinct model architectures to showcase different NLP design philosophies:
  * **Autoencoding Pipeline:** Fine-tuning a bidirectional `bert-base-uncased` model using sequence classification heads.
  * **Autoregressive Pipeline:** Tuning a causal `distilgpt2` model using a shifted language modeling collator for text completion.
  * **Adversarial Pipeline:** Constructing a custom `TextGenerator` (LSTM + Gumbel-Softmax) and `TextDiscriminator` (LSTM + Linear Classifier) loop built completely from scratch in PyTorch.

### 3. Training Histories
* **Location:** Saved cell outputs within the interactive `.ipynb` file.
* **Mechanism:** Every model execution logs real-time metric progressions across all training epochs. This includes Cross-Entropy Loss and F1 metrics for BERT, evaluation Cross-Entropy Loss for GPT, and shifting Discriminator/Generator loss constraints alongside Discriminator Accuracy tracking for the Text-GAN framework.

### 4. Test Inferences
* **Location:** Located at the termination blocks of Parts 2, 3, and 4.
* **Mechanism:** * BERT runs evaluation inferences on the out-of-sample `test_data` to generate its classification report.
  * GPT infers next-token sequences over the test subset to calculate an analytical Perplexity validation score.
  * The Text-GAN pipeline features a dedicated test inference phase that takes randomized Gaussian noise vectors, pushes them through the trained Generator network, maps the outputs back to English characters via an inverse index lookup dictionary, and prints concrete `Sample Generated Texts` to evaluate syntactic coherence.

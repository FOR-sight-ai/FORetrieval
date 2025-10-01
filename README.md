## Getting started

Currently, we support all models supported by the underlying [colpali-engine](https://github.com/illuin-tech/colpali), including the newer, and better, ColQwen2 checkpoints, such as `vidore/colqwen2-v1.0`.

Additional backends will be supported in future updates. As byaldi exists to facilitate the adoption of multi-modal retrievers, we intend to also add support for models such as [VisRAG](https://github.com/openbmb/visrag).

### Pre-requisites

#### Poppler

To convert pdf to images with a friendly license, we use the `pdf2image` library. This library requires `poppler` to be installed on your system. Poppler is very easy to install by following the instructions [on their website](https://poppler.freedesktop.org/). The tl;dr is:

__MacOS with homebrew__

```bash
brew install poppler
```

__Debian/Ubuntu__

```bash
sudo apt-get install -y poppler-utils
```

#### Flash-Attention

Gemma uses a recent version of flash attention. To make things run as smoothly as possible, we'd recommend that you install it after installing the library:

```bash
pip install flash-attn
```

#### Hardware

ColPali uses multi-billion parameter models to encode documents. We recommend using a GPU for smooth operations, though weak/older GPUs are perfectly fine! Encoding your collection would suffer from poor performance on CPU or MPS.

## Acknowledgements

FORetrieval was forked from Byaldi, [RAGatouille](https://github.com/answerdotai/ragatouille)'s mini sister project. It is a simple wrapper around the [ColPali](https://github.com/illuin-tech/colpali) repository to make it easy to use late-interaction multi-modal models such as ColPALI with a familiar API.
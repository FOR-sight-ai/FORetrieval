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

## Running the test suite

Install the dev dependencies first:

```bash
pip install -e ".[dev]"
# or with uv:
uv sync --extra dev
```

### Unit tests

No API keys, no GPU required — runs in seconds:

```bash
pytest -m "not slow and not integration"
```

### Metadata tests (no AI)

Tests for `ai_metadata_provider_factory(None)` and `build_metadata_list_for_dir`.  
No API key or GPU required:

```bash
pytest tests/test_metadata_no_ai.py
```

### Metadata tests (with AI)

Tests for AI-backed metadata generation .  
Set at least one of the following environment variables before running:

```bash
export OPENROUTER_API_KEY=...   
export OPENAI_API_KEY=...
export MISTRAL_API_KEY=...
export OLLAMA_HOST=http://localhost:11434   # + optionally OLLAMA_MODEL (default: mistral-small-latest)
```

All available backends are detected automatically and the suite runs once per backend:

```bash
pytest tests/test_metadata_ai.py -v
```

### Slow tests (GPU-dependent)

Tests that load a real ColPali model and perform indexing/search require a GPU and are gated behind `@pytest.mark.slow`:

```bash
pytest -m slow
```

### Markers reference

| Marker | Meaning |
|--------|---------|
| `slow` | GPU-dependent or computationally expensive |
| `integration` | Requires a live API key or Ollama daemon |

## Acknowledgements

FORetrieval was forked from Byaldi, [RAGatouille](https://github.com/answerdotai/ragatouille)'s mini sister project. It is a simple wrapper around the [ColPali](https://github.com/illuin-tech/colpali) repository to make it easy to use late-interaction multi-modal models such as ColPALI with a familiar API.

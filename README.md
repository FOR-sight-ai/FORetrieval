# FORetrieval

FORetrieval is a multimodal document retrieval library built on top of [colpali-engine](https://github.com/illuin-tech/colpali). It indexes document pages as images using late-interaction models (ColPali, ColQwen2, ColQwen2.5) and retrieves the most relevant pages for a given query. It is used by [FORag](https://github.com/FOR-sight-ai/FORAG) as its retrieval backend.

Key features:

- **Two storage backends** — local file-based (Colpali legacy `.pt` files) or Qdrant embedded vector store (default)
- **Metadata generation** — filesystem metadata always; AI-generated tags, language detection, and short descriptions optionally
- **Metadata filtering** — filter the retrieval pool by `ext`, `mtime`, `language`, `tags`, `document_type`, or arbitrary regex patterns before scoring
- **Docling ingestion** — optional semantic PDF chunking using [Docling](https://github.com/DS4SD/docling), producing image chunks aligned with document structure
- **Heatmap and circle visualisation** — relevance overlays for retrieved pages

## Installation

```bash
uv sync

# Optional extras:
uv sync --extra qdrant          # Qdrant storage backend (recommended for large indexes)
uv sync --extra docling         # Docling-based PDF chunking
uv sync --extra embedding_server  # Remote vLLM embedding server (adds paramiko for auto-deploy)
uv sync --extra quantization    # 4-bit / 8-bit local model quantization (adds bitsandbytes)
```

## Pre-requisites

### Poppler

Required by `pdf2image` for PDF-to-image conversion:

**Debian / Ubuntu**
```bash
sudo apt-get install -y poppler-utils
```

### Flash-Attention (optional)

Speeds up ColQwen2 / Gemma-based models significantly:

```bash
uv pip install flash-attn
```

### Hardware

ColPali uses multi-billion parameter models. A GPU is strongly recommended for indexing and search. Weak or older GPUs (sm_70+) work fine; CPU is supported but slow.

## Quick usage

```python
from foretrieval import MultiModalRetrieverModel

# Index a folder of PDFs
model = MultiModalRetrieverModel.from_pretrained(
    "vidore/colqwen2.5-v0.2",
    index_root="my_indexes",
    storage_qdrant=True,   # use Qdrant backend (default)
)
model.index(
    input_path="path/to/docs/",
    index_name="my_index",
    store_collection_with_index=True,
)

# Load an existing index and search
model = MultiModalRetrieverModel.from_index(
    index_path="my_index",
    index_root="my_indexes",
)
results = model.search("maximum output current", k=3)
for r in results:
    print(r.doc_id, r.page_num, r.score)
```

## Storage backends

FORetrieval supports two backends for storing embeddings:

| Backend | Constructor flag | Description |
|---------|-----------------|-------------|
| **Qdrant** (default) | `storage_qdrant=True` | Embeddings stored in a local embedded Qdrant database under `<index_root>/<index_name>/qdrant/`. Does not load all embeddings into RAM. Requires `foretrieval[qdrant]`. |
| **Local** | `storage_qdrant=False` | Embeddings saved as `.pt` files, loaded into memory at search time. No extra dependency. |

When loading an existing index with `from_index()`, the backend is read automatically from the saved `index_config.json.gz` — no manual flag needed.

```python
# Create with Qdrant backend
model = MultiModalRetrieverModel.from_pretrained(..., storage_qdrant=True)

# Create with local backend
model = MultiModalRetrieverModel.from_pretrained(..., storage_qdrant=False)

# Load existing index — backend auto-detected
model = MultiModalRetrieverModel.from_index(index_path="my_index", index_root=".")
```

## Metadata generation

Metadata can be attached to each document at indexing time. Two levels are available:

**Filesystem metadata (no AI required):** always populated from the file itself.

| Field | Source |
|-------|--------|
| `stem`, `ext`, `mime` | filename and MIME type |
| `mtime` | file modification time (ISO-8601 UTC) |
| `page_count` | number of pages (PDFs only) |
| `author`, `title` | embedded PDF metadata (may be absent) |
| `image_width`, `image_height` | dimensions (images only) |

**AI-generated metadata (requires an LLM provider):** `language`, `tags`, `document_type`, `short_description`.

```python
from foretrieval.metadata import ai_metadata_provider_factory
from foretrieval.models_metadata import build_metadata_list_for_dir

# No-AI provider: filesystem fields only
provider = ai_metadata_provider_factory(None)

# AI provider: enriches with language, tags, document_type, short_description
provider = ai_metadata_provider_factory({
    "provider": "openrouter",
    "name": "mistralai/mistral-small-3.2-24b-instruct",
    "api_key": "...",
})

metadata_list = build_metadata_list_for_dir(Path("docs/"), provider)

model.index(
    input_path="docs/",
    index_name="my_index",
    metadata=metadata_list,
)
```

## Metadata filtering

When an index was built with metadata, `search()` accepts a `filter_metadata` dict that restricts the scoring pool to matching documents only.

### Declared filter fields

```python
from foretrieval.models_metadata import MetadataFilter

# Only PDF files
results = model.search("max current", k=3, filter_metadata={"ext": ".pdf"})

# Files modified after a date
results = model.search("max current", k=3, filter_metadata={
    "mtime": {">=": "2025-01-01T00:00:00Z"}
})

# Multiple criteria (AND by default)
results = model.search("max current", k=3, filter_metadata={
    "ext": ".pdf",
    "language": "en",
})

# OR logic
results = model.search("max current", k=3, filter_metadata={
    "ext": [".pdf", ".docx"],
    "logic": "OR",
})
```

| Filter field | Type | Description |
|-------------|------|-------------|
| `ext` | `str` or `list[str]` | File extension(s) |
| `mtime` | `dict` | Operators: `>=`, `<=`, `>`, `<`, `==` against ISO-8601 string |
| `language` | `str` or `list[str]` | Language code(s), e.g. `"en"` |
| `tags` | `str` or `list[str]` | Any tag in common (requires AI metadata) |
| `document_type` | `str` or `list[str]` | Document type (requires AI metadata) |
| `logic` | `"AND"` or `"OR"` | How to combine criteria (default: `"AND"`) |

Any other key is matched by exact string equality against the stored metadata dict.

### Regex pattern matching

Use the `regex` field for substring or pattern matching on any text field. Patterns use Python `re.search` and are **always case-insensitive**:

```python
# Files whose name contains "general"
results = model.search("max current", k=3, filter_metadata={
    "regex": {"stem": "general"}
})

# Title contains "motor" or "pump"
results = model.search("specs", k=3, filter_metadata={
    "regex": {"title": "motor|pump"}
})

# Combine with ext filter
results = model.search("specs", k=3, filter_metadata={
    "ext": ".pdf",
    "regex": {"stem": "^report_2025"},
})
```

When the filter matches no documents, `search()` returns an empty list `[]` without raising.

## Docling ingestion

FORetrieval optionally uses [Docling](https://github.com/DS4SD/docling) to convert PDFs into semantically meaningful image chunks rather than whole pages. Each chunk corresponds to a coherent region of text and associated figures.

```python
model = MultiModalRetrieverModel.from_pretrained(
    "vidore/colqwen2.5-v0.2",
    ingestion={"backend": "docling"},
    index_root="my_indexes",
)
model.index(input_path="docs/", index_name="chunked_index")
```

Results include a `chunk_num` field identifying the exact Docling chunk within the page.

## Running the test suite

Install the dev dependencies first:

```bash
uv sync --extra dev
```

### Unit tests

No API keys, no GPU required — runs in seconds:

```bash
pytest -m "not slow and not integration"
```

### Metadata tests (no AI)

```bash
pytest tests/test_metadata_no_ai.py
```

### Metadata tests (with AI)

Set at least one API key:

```bash
export OPENROUTER_API_KEY=...
export OPENAI_API_KEY=...
export MISTRAL_API_KEY=...
export OLLAMA_HOST=http://localhost:11434   # + optionally OLLAMA_MODEL (default: mistral-small-latest)
```

```bash
pytest tests/test_metadata_ai.py -v
```

All available backends are detected automatically and the suite runs once per backend.

### Qdrant backend tests

```bash
# Unit tests (no GPU needed, Qdrant mocked)
pytest tests/test_qdrant.py -m "not slow and not integration"

# Full integration test (GPU + qdrant-client required)
pytest tests/test_qdrant.py -m "slow and integration"
```

### Metadata filter tests

```bash
pytest tests/test_metadata_filter.py
```

### Slow tests (GPU-dependent)

Full ColPali indexing and search:

```bash
pytest -m slow
```

### Markers reference

| Marker | Meaning |
|--------|---------|
| `slow` | GPU-dependent or computationally expensive |
| `integration` | Requires a live API key or Ollama daemon |

## Remote embedding server

FORetrieval can offload all embedding computation to a remote GPU server running [vLLM](https://docs.vllm.ai). The local machine only loads the processor (tokenizer + image preprocessor) — no model weights, no GPU required locally.

**Requirements:**
- vLLM ≥ 0.19.0 on the remote server
- Only **ColQwen3 / ColQwen3.5** models are supported by the vLLM `/pooling` endpoint. ColPali, ColQwen2, and ColQwen2.5 are not supported.
- Recommended model: `athrael-soju/colqwen3.5-4.5B-v3` (rank 3 on ViDoRe V3, 320-dim, Apache 2.0)

### Quick start

```python
from foretrieval import MultiModalRetrieverModel
from foretrieval.embedding_server import EmbeddingServerConfig

cfg = EmbeddingServerConfig(
    url="http://gpu-server:8000",
    model_name="athrael-soju/colqwen3.5-4.5B-v3",
)

model = MultiModalRetrieverModel.from_pretrained(
    "athrael-soju/colqwen3.5-4.5B-v3",
    index_root="my_indexes",
    embedding_server=cfg,
)
model.index("path/to/docs/", index_name="my_index")
results = model.search("maximum altitude", k=3)
```

### Auto-deploy

Set `auto_deploy=True` to have FORetrieval SSH to the GPU server and start the vLLM Docker container automatically if it is not already running. Requires `foretrieval[embedding_server]` (adds `paramiko`).

```python
cfg = EmbeddingServerConfig(
    url="http://gpu-server:8000",
    model_name="athrael-soju/colqwen3.5-4.5B-v3",
    auto_deploy=True,
    ssh_host="gpu-server",       # SSH target
    ssh_user="myuser",           # optional, defaults to $USER
    n_gpus=-1,                   # -1 = all available GPUs (auto-detected via nvidia-smi)
)
```

The manager pulls `vllm/vllm-openai:latest`, starts the container with `--tensor-parallel-size N`, and writes a metadata file at `~/.foretrieval/deployment.json` on the remote. Subsequent calls detect the running container and skip redeployment.

### Authentication and SSL

```python
cfg = EmbeddingServerConfig(
    url="https://gpu-server:8000",
    model_name="athrael-soju/colqwen3.5-4.5B-v3",
    api_key="my-secret-token",   # Authorization: Bearer header
    verify_ssl=False,            # for self-signed certificates
)
```

Deploy vLLM with `--api-key my-secret-token` to require authentication.

### SSH tunnel (firewalled servers)

If port 8000 is not directly reachable, open an SSH tunnel first:

```bash
ssh -fNL 8000:localhost:8000 gpu-server
```

Then use `http://localhost:8000` as the URL.

### EmbeddingServerConfig reference

| Field | Default | Description |
|-------|---------|-------------|
| `url` | required | Base URL of the vLLM server |
| `model_name` | required | HuggingFace model ID (must contain `colqwen3`) |
| `auto_deploy` | `false` | SSH + Docker auto-deploy |
| `ssh_host` | `None` | SSH hostname (required when `auto_deploy=True`) |
| `ssh_user` | `None` | SSH username (defaults to `$USER`) |
| `ssh_key_path` | `None` | Path to SSH private key (defaults to SSH agent) |
| `n_gpus` | `-1` | Number of GPUs (`-1` = all available) |
| `port` | `8000` | Port exposed on the remote server |
| `hf_token` | `None` | HuggingFace token for gated models |
| `api_key` | `None` | Bearer token for server authentication |
| `verify_ssl` | `True` | Verify SSL certificates |
| `batch_size` | `4` | Images per request (auto-halved on OOM) |
| `request_timeout` | `120` | HTTP timeout in seconds |

## Local model quantization

For local (non-remote) inference, 4-bit and 8-bit quantization reduce VRAM usage via [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes). Requires `foretrieval[quantization]` and a CUDA device.

```python
model = MultiModalRetrieverModel.from_pretrained(
    "vidore/colqwen2.5-v0.2",
    load_in_4bit=True,                  # or load_in_8bit=True
    bnb_4bit_quant_type="nf4",          # "nf4" (default) or "fp4"
    bnb_4bit_compute_dtype="float16",   # compute dtype
)
```

## Acknowledgements

FORetrieval was originally forked from [Byaldi](https://github.com/answerdotai/byaldi), a wrapper around the [ColPali](https://github.com/illuin-tech/colpali) repository. It has since diverged significantly to add metadata generation and filtering, Qdrant storage, Docling ingestion, and heatmap visualisation.

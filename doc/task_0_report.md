# Task 0 — Embedding Server for FORetrieval and FORag

## Summary

Implementation of a remote vLLM-based embedding server for FORetrieval, with client integration and FORag wiring. ColQwen3.5 replaces ColQwen2.5 as the default retrieval model.

---

## Step 1 — Server implementation candidates

**Candidates evaluated:**

| Option | Pros | Cons |
|---|---|---|
| **vLLM** (chosen) | Native ColPali/ColQwen3.5 support (`ColQwen3_5` architecture), multi-GPU tensor parallelism built-in, production-grade server, `/pooling` endpoint for token-level embeddings | Large Docker image (~20 GB) |
| Custom FastAPI + colpali-engine | Matches existing local code exactly, smaller image | No built-in multi-GPU, no batching optimisation, more code to write and maintain |

**Decision:** vLLM, Docker deployment, server-side multi-GPU via `--tensor-parallel-size`.

**Inputs:** task_0.md requirements, vLLM docs, colpali-engine release notes.
**Outputs:** architecture decision documented here.

---

## Step 2 — Branch creation

Two `embedding_server` branches created:

```bash
git -C FORetrieval checkout -b embedding_server
git -C FORAG     checkout -b embedding_server
```

---

## Step 3 — colpali-engine upgrade: ColQwen3_5 support

`ColQwen3_5` and `ColQwen3_5Processor` were added in **colpali-engine 0.3.15** (released 31 Mar 2026). Both repos bumped:

| Repo | Old constraint | New constraint |
|---|---|---|
| FORetrieval | `>=0.3.4,<0.4.0` | `>=0.3.15,<0.4.0` |
| FORAG | `>=0.3.12` | `>=0.3.15` |

**`colpali.py` changes:**
- Model name validation updated to also accept `colqwen3` names.
- `_resolve_model_and_processor_classes()` helper added with dispatch:
  - `colpali` → `ColPali` / `ColPaliProcessor`
  - `colqwen3.5` / `colqwen3_5` → `ColQwen3_5` / `ColQwen3_5Processor`
  - `colqwen2.5` → `ColQwen2_5` / `ColQwen2_5_Processor`
  - fallback → `ColQwen2` / `ColQwen2Processor`

**Default model in smartcockpit config:** updated from `vidore/colqwen2.5-v0.2` → `athrael-soju/colqwen3.5-4.5B-v3` (rank 3 on ViDoRe V3, **320-dim**, 4.5B params, Apache 2.0).

---

## Step 4 — Server package: `foretrieval/embedding_server/`

New package with three modules:

### `config.py` — `EmbeddingServerConfig`
Pydantic model holding all server parameters:
- `url`: base URL of the vLLM server
- `model_name`: HuggingFace model ID
- `auto_deploy`: explicit opt-in flag for SSH-based deployment
- `ssh_host`, `ssh_user`, `ssh_key_path`: SSH credentials
- `n_gpus`: -1 = all available GPUs (detected via `nvidia-smi` over SSH)
- `port`, `hf_token`, `batch_size`, `request_timeout`

Validation: `auto_deploy=True` requires `ssh_host`; `n_gpus` cannot be 0.

### `client.py` — `EmbeddingServerClient`
HTTP client for the vLLM `/pooling` endpoint:
- `embed_images(images)` → `List[Tensor[n_tokens, 320]]` — images sent as base64 PNG via `PoolingChatRequest` (messages array)
- `embed_query(query)` → `List[Tensor[n_tokens, 320]]` — text query via flat `input` field
- `health_check()` → `bool`

**vLLM 0.19.0 image format:** images must use `PoolingChatRequest` (`messages` array with `image_url` content). The flat `input` field only accepts token id lists and rejects image dicts with HTTP 400.

**OOM handling:** wraps each batch in a retry loop. On HTTP 500 with OOM markers (`"CUDA out of memory"`, `"OOM"`, etc.), batch size is halved and the request retried. Minimum batch size is 1; if OOM persists at 1, `ServerOOMError` is raised.

### `manager.py` — `EmbeddingServerManager`
SSH-based Docker deployment manager:
1. Read `~/.foretrieval/deployment.json` on remote (metadata file).
2. If absent → deploy from scratch (pull image, run container, write metadata).
3. If present → health-check via `docker inspect`; redeploy if stopped.
4. `n_gpus=-1` → auto-detect via `nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`.
5. Uses `paramiko` (optional dep under `[embedding_server]` extra).

Docker command (vLLM 0.19.0, image entrypoint = `vllm serve`):
```bash
docker run -d \
  --name foretrieval_embedding_server \
  --gpus all \                           # or --gpus N if n_gpus > 0
  -p <port>:8000 \
  -e HF_HOME=/opt/huggingface \
  -v /opt/huggingface:/opt/huggingface \
  --restart unless-stopped --ipc=host \
  vllm/vllm-openai:latest \
  <model_name> \
  --runner pooling \
  --convert embed \
  --tensor-parallel-size <n_gpus> \
  --gpu-memory-utilization 0.7 \
  --max-model-len 8192 \
  --trust-remote-code
```

**Note:** `--task token_embed` was removed in vLLM 0.19.0. Use `--runner pooling --convert embed` instead. `--max-model-len 8192` caps the encoder cache budget to prevent OOM during profiling on 24 GB GPUs (default 262144 exceeds available VRAM).

Remote metadata file (`~/.foretrieval/deployment.json`):
```json
{
  "model_name": "...",
  "container_name": "foretrieval_embedding_server",
  "port": 8000,
  "n_gpus": 2,
  "image": "vllm/vllm-openai:latest",
  "deployed_at": "2026-04-15T..."
}
```

---

## Step 5 — ColPaliModel remote embedding integration

**`colpali.py` changes:**
- `__init__` accepts new `embedding_server: Optional[EmbeddingServerConfig]` parameter.
- When set:
  - If `auto_deploy=True`: calls `EmbeddingServerManager.ensure_deployed()`.
  - Instantiates `EmbeddingServerClient`.
  - Calls `_load_processor_only()` instead of `_load_model_and_processor()` — processor only (CPU), no model weights loaded on the client machine. This saves ~9 GB VRAM locally.
- `_add_to_index`: processor runs locally (CPU) for heatmap sidecar data (`input_ids`, `image_grid_thw`, `orig_sizes`); embedding computation dispatched to remote client instead of local GPU.
- `_encode_search_query`: dispatched to remote client when active; otherwise existing local GPU path unchanged.
- `from_pretrained` / `from_index`: `embedding_server` param threaded through.

**Backward compatibility:** no `embedding_server` in config → identical behaviour to before.

**Heatmaps:** fully preserved. Processor runs locally (CPU-only) to generate sidecar data even in remote mode.

---

## Step 6 — retriever.py and pipeline.py wiring

**`retriever.py`**: `MultiModalRetrieverModel.from_pretrained` and `from_index` both accept and pass through `embedding_server: Optional[EmbeddingServerConfig]`.

**`forag/pipeline.py` `_create_retriever`**: reads optional `retriever.embedding_server` dict from config, constructs `EmbeddingServerConfig.from_dict(...)`, passes to `MultiModalRetrieverModel`.

---

## Step 7 — Config changes

**`FORAG/config/schema.json`**: added optional `embedding_server` object to retriever schema with all fields documented.

**`FORAG/config/smartcockpit_openrouter.json`**: model updated to ColQwen3.5; `embedding_server` block added with placeholder host `<GPU_SERVER_HOST>`.

---

## Step 8 — Dependencies

**FORetrieval `pyproject.toml`**:
- Core: added `requests>=2.32` (HTTP client)
- New optional group: `[embedding_server]` = `paramiko>=3.0` (SSH for auto-deploy)
- Dev: added `pytest-mock>=3.0`

---

## Step 9 — Tests

### Unit tests (offline, no GPU, no SSH)

**`FORetrieval/tests/test_embedding_server.py`** — 33 tests covering:
- `EmbeddingServerConfig` validation (n_gpus, auto_deploy, trailing slash, from_dict)
- `EmbeddingServerClient` request format (payload structure, endpoint, timeout)
- `EmbeddingServerClient` response parsing (tensor shape, dtype, count)
- OOM retry logic (halving, batch_size=1 failure, non-OOM 500, connection/timeout errors)
- Health check
- `EmbeddingServerManager` deploy/skip/redeploy logic, GPU detection, stop, missing paramiko

**`FORAG/tests/test_pipeline_embedding_server.py`** — 5 tests covering:
- No `embedding_server` key → `None` passed to retriever
- `embedding_server` block → `EmbeddingServerConfig` constructed and passed to `from_pretrained` / `from_index`
- `auto_deploy` defaults to False
- Missing required field raises validation error

**Results:**
```
FORetrieval: 100 passed, 5 skipped (qdrant optional dep), 5 deselected (slow/integration)
FORAG:       120 passed, 7 pre-existing failures (mistral optional dep not installed)
```

### Integration tests

**`FORetrieval/tests/test_embedding_server_integration.py`** — driven by environment variables:
```
FORETRIEVAL_TEST_SERVER=http://<GPU_SERVER_HOST>:8000
FORETRIEVAL_TEST_SSH_HOST=<GPU_SERVER_HOST>
FORETRIEVAL_TEST_MODEL=athrael-soju/colqwen3.5-4.5B-v3  # optional
```
All tests skipped if `FORETRIEVAL_TEST_SERVER` not set. Server not directly reachable from client network: use an SSH tunnel.

```bash
# Open tunnel (local 18000 → remote 8000)
ssh -N -L 18000:localhost:8000 <GPU_SERVER_HOST> &

# Run tests
FORETRIEVAL_TEST_SERVER=http://localhost:18000 \
FORETRIEVAL_TEST_SSH_HOST=<GPU_SERVER_HOST> \
uv run pytest tests/test_embedding_server_integration.py -m "integration" -v
```

**Final results (live server, 2× RTX4090, `athrael-soju/colqwen3.5-4.5B-v3`):**
```
11 passed, 2 deselected (manager SSH deploy tests)
```

Covers: health check, single/multi image embedding, query embedding, tensor shapes (320-dim confirmed), CPU placement, batch embedding, OOM retry path, `ColPaliModel` end-to-end remote indexing and query encoding.

---

## Step 10 — Remote server deployment

**Server:** 2× RTX4090 (24 GB each), Ubuntu 22.04, Docker 29.4.0, vLLM 0.19.0.

**Prerequisites installed by sysadmin:**
- Docker 29.4.0 (added via `apt`)
- nvidia-container-toolkit (added via NVIDIA apt repo — `https://nvidia.github.io/libnvidia-container`)

**Convenience script** (`~/start_embedding_server.sh` on remote):
```bash
bash ~/start_embedding_server.sh
```
Auto-detects GPU count, starts container, polls `/health` for up to 300s, writes `~/.foretrieval/deployment.json`.

**Actual docker run command used:**
```bash
docker run -d \
  --name foretrieval_embedding_server \
  --gpus all \
  -p 8000:8000 \
  -e HF_HOME=/opt/huggingface \
  -v /opt/huggingface:/opt/huggingface \
  --restart unless-stopped --ipc=host \
  vllm/vllm-openai:latest \
  athrael-soju/colqwen3.5-4.5B-v3 \
  --runner pooling \
  --convert embed \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 8192 \
  --trust-remote-code
```

**Startup time:** ~165s (model load + NCCL init + piecewise CUDA graph compilation).

**Memory usage at steady state:** ~9 GB per GPU (model sharded across 2 GPUs via TP=2).

---

## Encountered difficulties

| Issue | Resolution |
|---|---|
| `ColQwen3_5` not in `colpali-engine<0.3.15` | Bumped lower bound to `>=0.3.15` |
| `nvidia-container-toolkit` not in default Ubuntu apt repos | Sysadmin added NVIDIA apt repo and installed via `apt` |
| `apt` reported "unable to locate package nvidia-container-toolkit" initially | Added NVIDIA container toolkit apt repo first |
| vLLM 0.19.0: `--task token_embed` flag removed | Replaced with `--runner pooling --convert embed` |
| vLLM 0.19.0: flat `"input"` rejects image dicts (HTTP 400) | Images sent via `PoolingChatRequest` (`messages` array) instead |
| ColQwen2/ColQwen2.5 not supported in vLLM 0.19.0 | Only ColQwen3 and ColQwen3_5 supported; used `athrael-soju/colqwen3.5-4.5B-v3` |
| ColQwen3.5 OOM during KV cache init (default `max_seq_len=262144`) | Added `--max-model-len 8192` and `--gpu-memory-utilization 0.7` |
| vLLM Docker image ~32 GB, slow remote pull | Ran `docker pull` detached; user pulled manually |
| Remote server port 8000 not reachable from client network (firewall) | SSH tunnel: `ssh -N -L 18000:localhost:8000 <host>` |
| `image_grid_thw` absent for ColPali (only ColQwen) | Guarded sidecar writes with `if grid_cpu is not None` |
| ColQwen3.5 embed dim is 320 not 128 | Corrected dim assertions in integration tests |
| `test_metadata_ai.py` / `test_integration.py` collection error (pre-existing, missing API key env vars) | Excluded from offline test run; unrelated to this task |
| 7 pre-existing FORAG test failures (`mistral` optional dep not installed) | Pre-existing; unrelated to this task |

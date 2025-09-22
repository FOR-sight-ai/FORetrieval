import argparse
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import coloredlogs
from jsonschema import ValidationError, validate

from FORetrieval import MultiModalRetrieverModel
from forag.agents import (
    AgenticRetriever,
    AgenticGenerator,
    Digest,
    DigestEntry,
)
from forag.generator import Generator


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #
def setup_logging(log_path: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure ``log_path`` and ``coloredlogs``.
    Returns the configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger("forag")
    logger.setLevel(level)
    # Silence noisy third‑party libraries
    for name in ["torch", "transformers", "httpx", "LiteLLM", "colpali_engine"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    coloredlogs.install(
        logger=logger,
        fmt=fmt,
        datefmt="%Y/%m/%d %H:%M:%S",
        level=level,
        programname="forag",
        field_styles={
            "levelname": {"color": "blue", "bold": True},
            "asctime": {"color": "yellow"},
            "name": {"color": "green"},
        },
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)
    return logger


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #
def load_config(config_file_path: Path) -> Dict[str, Any]:
    """Load a JSON configuration file."""
    try:
        with open(config_file_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {config_file_path}")
    except json.JSONDecodeError as exc:
        logging.error(f"Failed to parse JSON: {exc}")
    return {}


def validate_json_data(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Validate data against a JSON‑schema."""
    try:
        validate(instance=data, schema=schema)
    except ValidationError as exc:
        logging.error(f"Configuration validation error: {exc}")


# --------------------------------------------------------------------------- #
# Component constructors – fully typed.
# Each function works on a **deep copy** of the config fragment it needs
# to keep the original config side‑effect free.
# --------------------------------------------------------------------------- #
def get_retriever(config: Dict[str, Any]) -> MultiModalRetrieverModel:
    """
    Construct/ load the ``MultiModalRetrieverModel``.
    The function respects the ``overwrite`` flag – if the index does not
    exist or ``overwrite`` is true the index will be rebuilt.
    """
    cfg = copy.deepcopy(config)
    retr_cfg = cfg["retriever"]
    model_cfg = retr_cfg["model"]
    index_cfg = retr_cfg["index"]
    name = index_cfg["name"]
    root = index_cfg["root"]
    data_path = Path(index_cfg["data_path"]).resolve()
    overwrite = index_cfg["overwrite"]
    pointer_based = index_cfg["pointer_based"]
    if not Path(index_cfg["data_path"]).exists() or overwrite:
        rag = MultiModalRetrieverModel.from_pretrained(
            name=model_cfg["name"],
            index_root=root,
            device=model_cfg["device"],
            verbose=cfg.get("verbose", False),
        )
        rag.index(
            input_path=data_path,
            index_name=name,
            store_collection_with_index=not pointer_based,
            overwrite=overwrite,
            **{
                k: v
                for k, v in index_cfg.items()
                if k not in {"root", "data_path", "name", "overwrite", "pointer_based"}
            },
        )
    else:
        rag = MultiModalRetrieverModel.from_index(
            index_path=name,
            index_root=root,
            device=model_cfg["device"],
            verbose=cfg.get("verbose", False),
        )
    return rag


def get_generator(config: Dict[str, Any]) -> Generator:
    """
    Build a classic ``Generator`` (retrieval → generation) instance.
    The returned generator has no ``digest``; it will be injected later.
    """
    cfg = copy.deepcopy(config)
    gen_cfg = cfg["generator"]["model"]
    return Generator.get_generator(
        backend=gen_cfg["provider"],
        model_name=gen_cfg["name"],
        api_key=gen_cfg.get("api_key"),
        digest=None,  # To be attached later
        **{
            k: v for k, v in gen_cfg.items() if k not in {"provider", "name", "api_key"}
        },
    )


def get_agentic_retriever(
    config: Dict[str, Any], rag: MultiModalRetrieverModel
) -> AgenticRetriever:
    """
    Build the agentic retriever that delegates the heavy lifting to the
    already‑constructed ``rag`` instance.
    """
    cfg = copy.deepcopy(config)
    agen_cfg = cfg["agents"]["retriever"]
    model_cfg = agen_cfg["model"]
    retriever = AgenticRetriever.get_agentic_retriever(
        backend=model_cfg["provider"],
        model_name=model_cfg["name"],
        image_analyzer_params=agen_cfg["image_analyzer"]["model"],
        retriever=rag,
        url=model_cfg.get("url", ""),
        api_key=model_cfg.get("api_key", ""),
        max_steps=agen_cfg["max_steps"],
        **{
            k: v
            for k, v in agen_cfg.items()
            if k not in {"model", "image_analyzer", "max_steps"}
        },
        **{
            k: v
            for k, v in model_cfg.items()
            if k not in {"provider", "name", "url", "api_key"}
        },
    )
    return retriever


def get_agentic_generator(
    config: Dict[str, Any],
    rag: MultiModalRetrieverModel,
    agentic_retriever: AgenticRetriever,
    digest: Digest,
) -> AgenticGenerator:
    """
    Build the agentic generator; attaches ``rag`` & ``digest`` later.
    """
    cfg = copy.deepcopy(config)
    agen_cfg = cfg["agents"]["generator"]
    img_cfg = cfg["agents"]["retriever"]["image_analyzer"]
    generator = AgenticGenerator(
        backend=agen_cfg["model"]["provider"],
        model_name=agen_cfg["model"]["name"],
        max_steps=agen_cfg["max_steps"],
        retrieval_agent=agentic_retriever,
        image_analyzer_params=img_cfg["model"],
        retriever=rag,
        digest=digest,
        **{k: v for k, v in agen_cfg["model"].items() if k not in {"provider", "name"}},
    )
    return generator


def get_digest(config: Dict[str, Any]) -> Digest:
    """
    Build the digest container once.
    """
    cfg = copy.deepcopy(config)
    digest_cfg = cfg["digest"]
    model_cfg = digest_cfg["model"]
    return Digest.get_digestor(
        max_entries=digest_cfg["max_entries"],
        max_size=digest_cfg["max_size"],
        backend=model_cfg["provider"],
        model_name=model_cfg["name"],
        **{k: v for k, v in model_cfg.items() if k not in {"provider", "name"}},
    )


# --------------------------------------------------------------------------- #
# Generation helpers – they use the already‑constructed objects.
# --------------------------------------------------------------------------- #
def classical_generation(
    rag: MultiModalRetrieverModel,
    query: str,
    top_k: int,
    generator: Generator,
    digest: Digest,
) -> Tuple[str, str]:
    """
    Classical (retrieval → generation) pipeline.
    The ``generator`` instance is expected NOT to have a ``digest`` yet
    – we attach it before the generation step.
    """
    logging.debug("\nRetrieving\n")
    results = rag.search(query, k=top_k, return_base64_results=True)
    logging.debug(f"\nN Results: {len(results)}\n")
    logging.debug("\nGenerating response\n")
    generator.digest = digest
    response = generator(query, results)
    return query, response


def agentic_generation(
    query: str,
    agentic_generator: AgenticGenerator,
    digest: Digest,
) -> Tuple[str, str]:
    """
    Agentic RAG pipeline generator.
    """
    logging.info("\nAgentic Generation\n")
    agentic_generator.digest = digest
    response = agentic_generator(query)
    return query, response


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FORAG agentic RAG")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.json",
        help="Path to the config file",
    )
    parser.add_argument(
        "--agentic", action="store_true", default=False, help="Use agentic RAG"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in CLI mode (no Gradio). If omitted, a Gradio interface is launched.",
    )
    # <-- positional argument is now truly optional
    parser.add_argument(
        "query",
        nargs="?",  # <‑‑ makes it *optional*
        type=str,
        help="Initial query to run. Ignored when launching the Gradio UI.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser


def run_cli(query: str, args: argparse.Namespace) -> None:
    """
    The heavy lifting that the original `main()` did.
    Re‑used from the old interactive loop.
    """
    # Load configuration, set up logging, etc. – exactly as in `main()`
    config = load_config(Path(args.config))
    if not config:
        raise SystemExit("Configuration file could not be read or was invalid.")

    if not config.get("HF_TOKEN"):
        config["HF_TOKEN"] = os.getenv("HF_TOKEN", None)

    logger = setup_logging(
        config["log_path"], logging.DEBUG if args.verbose else logging.INFO
    )
    logger.info("Starting FORAG")
    config["verbose"] = args.verbose

    # Build heavy objects once
    rag = get_retriever(config)
    digest = get_digest(config)
    classical_generator = get_generator(config)
    classical_generator.digest = digest

    agentic_retriever: Optional[AgenticRetriever] = None
    agentic_generator: Optional[AgenticGenerator] = None
    if args.agentic:
        agentic_retriever = get_agentic_retriever(config, rag)
        if "generator" in config.get("agents", {}):
            agentic_generator = get_agentic_generator(
                config, rag, agentic_retriever, digest
            )

    # The interactive loop (kept identical to the original)
    top_k = config["retriever"]["retrieval"]["top_k"]
    current_query = query

    try:
        while True:
            if args.agentic:
                q, a = agentic_generation(
                    query=current_query,
                    agentic_generator=agentic_generator,
                    digest=digest,
                )
            else:
                q, a = classical_generation(
                    rag=rag,
                    query=current_query,
                    top_k=top_k,
                    generator=classical_generator,
                    digest=digest,
                )
            logger.info(a)
            nxt = input("Enter a follow‑up question or EXIT to exit: ")
            if nxt.strip().lower() == "exit":
                break
            digest.add(
                DigestEntry(
                    index=config["retriever"]["index"]["name"],
                    query=q,
                    answer=a,
                )
            )
            current_query = nxt
    except (KeyboardInterrupt, EOFError):
        logger.info("\nInterrupted – exiting.")


def run_gradio(args: argparse.Namespace) -> None:
    """
    Build a minimal, but fully‑powered Gradio interface.
    """
    import gradio as gr

    # -------- load / build stuff (shared with CLI) -------------
    config = load_config(Path(args.config))
    if not config:
        raise SystemExit("Configuration file could not be read or was invalid.")

    # No need to expose the logger in the UI, so we use INFO only.
    logger = setup_logging(config["log_path"], logging.INFO)
    logger.info("Launching Gradio UI")

    config["verbose"] = args.verbose

    rag = get_retriever(config)
    digest = get_digest(config)
    classical_generator = get_generator(config)
    classical_generator.digest = digest

    agentic_retriever: Optional[AgenticRetriever] = None
    agentic_generator: Optional[AgenticGenerator] = None
    if args.agentic:
        agentic_retriever = get_agentic_retriever(config, rag)
        if "generator" in config.get("agents", {}):
            agentic_generator = get_agentic_generator(
                config, rag, agentic_retriever, digest
            )

    top_k = config["retriever"]["retrieval"]["top_k"]

    # ---------- State objects that survive across invocations ----------
    # 1️⃣  history_state   – plain string that holds “Query → Answer” pairs.
    # 2️⃣  digest_state    – string representation of the Digest container.
    history_state = gr.State(value="")  # e.g. “Q1 → A1\nQ2 → A2"
    digest_state = gr.State(value="")  # JSON / pretty print

    # ------------------- helper function for Gradio -------------------
    def _answer(q: str) -> Tuple[str, str, str]:
        """
        Run a single query (classical or agentic) and return
        three things:
          * answer            – what the user sees in the “Answer” textbox
          * updated history   – the conversation log
          * updated digest    – the digest container as a string
        The function also mutates the persistent `history_state` and
        `digest_state` objects.
        """
        # ---- run the actual RAG pipeline ----
        if args.agentic:
            q, a = agentic_generation(
                query=q,
                agentic_generator=agentic_generator,
                digest=digest,
            )
        else:
            q, a = classical_generation(
                rag=rag,
                query=q,
                top_k=top_k,
                generator=classical_generator,
                digest=digest,
            )

        # ---- push an entry into the digest container ----
        digest.add(
            DigestEntry(
                index=config["retriever"]["index"]["name"],
                query=q,
                answer=a,
            )
        )

        # ---- update the conversation history (append) ----
        new_line = f"<span style='color: blue;'>**Q:** {q}</span><br><span style='color: green;'>**A:** {a}</span><br><span style='color: blue;'></span>"
        history_state.value = history_state.value + new_line + "\n"

        digest_state.value = repr(digest)

        # The function returns the three strings that will be fed to the UI
        return a, history_state.value, digest_state.value

    # -------------------------- Gradio UI --------------------------
    demo = gr.Blocks()
    with demo:
        gr.Markdown("<h1>FORAG web UI</h1>")

        # ── Question / Answer ─────────────────────────────
        with gr.Row():
            inp = gr.Textbox(
                label="Ask me anything",
                placeholder="Enter your question…",
                lines=1,
            )
            out_ans = gr.Textbox(
                label="Answer",
                lines=5,
                interactive=False,
                elem_id="answer-box",
            )

        # ── Conversation history and digest side by side ──────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                history_box = gr.Markdown(
                    label="Conversation History",
                    value="",  # will be filled by _answer
                    elem_id="history-box",
                )
            with gr.Column(scale=1):
                digest_box = gr.Markdown(
                    label="Digest",
                    value="",  # will be filled by _answer
                    elem_id="digest-box",
                )

        # ── Wiring ─────────────────────────────────────────────
        inp.submit(
            fn=_answer,
            inputs=[inp],
            outputs=[out_ans, history_box, digest_box],
        )

        # ── “Clear conversation” button -------------------------
        def _clear() -> Tuple[str, str, str]:
            """
            Reset history, digest, and UI boxes.  Returns empty strings
            for all outputs.
            """
            digest.clear()
            history_state.value = ""
            digest_state.value = ""
            return "", "", ""

        clear_btn = gr.Button("Clear conversation")
        clear_btn.click(
            fn=_clear, inputs=[], outputs=[out_ans, history_box, digest_box]
        )

    demo.launch(share=False)


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    # ────────────────────────────────────────────────────────────────────────
    #  1️⃣  If running in headless (CLI) mode, we *need* a query.
    #      Otherwise, fall back to the interactive prompt – the same logic
    #      that the original script used.
    if args.headless:
        # If the user omitted a query, prompt now.
        query = args.query or input("Enter your first question: ")
        run_cli(query, args)
    else:
        # ───── Gradio mode ────────
        # Completely ignore args.query – the UI will provide its own textbox.
        run_gradio(args)


################################################
if __name__ == "__main__":
    main()

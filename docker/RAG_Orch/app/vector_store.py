import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from pymilvus import DataType, MilvusClient
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)

from app.config import (
    MILVUS_PAGE_VECTOR_FIELD,
    MILVUS_PRIMARY_KEY_MAX_LENGTH,
    MILVUS_TOKEN_GROUP_FIELD,
    MILVUS_TOKEN_VECTOR_FIELD,
    PAYLOAD_JSON_MAX_LENGTH,
    get_settings,
    get_vector_db_backend,
)
from app.helpers import (
    deserialize_payload,
    mean_pool_vectors,
    milvus_collection_names,
    milvus_quote_string,
    page_identifier,
    qdrant_point_id,
    serialize_payload,
)


@dataclass
class PendingIndexBatch:
    qdrant_points: list[PointStruct] = field(default_factory=list)
    milvus_pages: list[dict[str, Any]] = field(default_factory=list)
    milvus_tokens: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SearchHit:
    point_id: str
    score: float
    payload: dict[str, Any]


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    # Create and cache the Qdrant client used by the orchestrator process.
    return QdrantClient(url=get_settings().qdrant_url)


@lru_cache(maxsize=1)
def get_milvus_client() -> MilvusClient:
    # Create and cache the Milvus client used by the orchestrator process.
    client_kwargs: dict[str, Any] = {"uri": get_settings().milvus_url}
    if get_settings().milvus_token:
        client_kwargs["token"] = get_settings().milvus_token
    return MilvusClient(**client_kwargs)


def ensure_qdrant_collection(collection_name: str, vector_size: int, recreate: bool) -> None:
    # Create or recreate a Qdrant multivector collection for ColQwen page storage.
    client = get_qdrant_client()

    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM,
                ),
            ),
        )


def milvus_collection_exists(client: MilvusClient, collection_name: str) -> bool:
    # Check whether a Milvus collection already exists.
    return collection_name in client.list_collections()


def ensure_milvus_collection_loaded(collection_name: str) -> None:
    # Load a Milvus collection in memory before inserts or searches.
    client = get_milvus_client()
    client.load_collection(collection_name=collection_name, replica_number=1)


def create_milvus_page_collection(collection_name: str, vector_size: int) -> None:
    # Create the Milvus page-level candidate collection used before reranking.
    client = get_milvus_client()

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=MILVUS_PRIMARY_KEY_MAX_LENGTH,
    )
    schema.add_field(
        field_name=MILVUS_PAGE_VECTOR_FIELD,
        datatype=DataType.FLOAT_VECTOR,
        dim=vector_size,
    )
    schema.add_field(
        field_name="payload_json",
        datatype=DataType.VARCHAR,
        max_length=PAYLOAD_JSON_MAX_LENGTH,
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=MILVUS_PAGE_VECTOR_FIELD,
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def create_milvus_token_collection(collection_name: str, vector_size: int) -> None:
    # Create the Milvus token-level collection used for late-interaction reranking.
    client = get_milvus_client()

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=MILVUS_PRIMARY_KEY_MAX_LENGTH,
    )
    schema.add_field(
        field_name=MILVUS_TOKEN_GROUP_FIELD,
        datatype=DataType.VARCHAR,
        max_length=MILVUS_PRIMARY_KEY_MAX_LENGTH,
    )
    schema.add_field(
        field_name=MILVUS_TOKEN_VECTOR_FIELD,
        datatype=DataType.FLOAT_VECTOR,
        dim=vector_size,
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=MILVUS_TOKEN_VECTOR_FIELD,
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def ensure_milvus_collections(collection_name: str, vector_size: int, recreate: bool) -> None:
    # Create or recreate the pair of Milvus collections required by the late-interaction layout.
    client = get_milvus_client()
    page_collection, token_collection = milvus_collection_names(collection_name)

    if recreate:
        if milvus_collection_exists(client, token_collection):
            client.drop_collection(collection_name=token_collection)
        if milvus_collection_exists(client, page_collection):
            client.drop_collection(collection_name=page_collection)

    if not milvus_collection_exists(client, page_collection):
        create_milvus_page_collection(page_collection, vector_size)
    if not milvus_collection_exists(client, token_collection):
        create_milvus_token_collection(token_collection, vector_size)

    ensure_milvus_collection_loaded(page_collection)
    ensure_milvus_collection_loaded(token_collection)


def ensure_collection(collection_name: str, vector_size: int, recreate: bool) -> None:
    # Ensure the logical collection exists on the configured backend before indexing begins.
    if get_vector_db_backend() == "qdrant":
        ensure_qdrant_collection(collection_name, vector_size, recreate)
        return
    ensure_milvus_collections(collection_name, vector_size, recreate)


def add_index_record(
    batch: PendingIndexBatch,
    backend: str,
    document_id: str,
    page_number: int,
    vectors: list[list[float]],
    payload: dict[str, Any],
) -> None:
    # Append one indexed page to the backend-specific in-memory batch structure.
    if backend == "qdrant":
        batch.qdrant_points.append(
            PointStruct(
                id=qdrant_point_id(),
                vector=vectors,
                payload=payload,
            )
        )
        return

    page_id = page_identifier(document_id, page_number)
    batch.milvus_pages.append(
        {
            "id": page_id,
            MILVUS_PAGE_VECTOR_FIELD: mean_pool_vectors(vectors),
            "payload_json": serialize_payload(payload),
        }
    )
    batch.milvus_tokens.extend(
        {
            "id": str(uuid.uuid4()),
            MILVUS_TOKEN_GROUP_FIELD: page_id,
            MILVUS_TOKEN_VECTOR_FIELD: vector,
        }
        for vector in vectors
    )


def pending_page_count(batch: PendingIndexBatch, backend: str) -> int:
    # Return the number of pending page records regardless of the configured backend.
    if backend == "qdrant":
        return len(batch.qdrant_points)
    return len(batch.milvus_pages)


def flush_qdrant_points(collection_name: str, points: list[PointStruct]) -> int:
    # Send the accumulated Qdrant points to the server and clear the local batch.
    if not points:
        return 0

    client = get_qdrant_client()
    client.upsert(collection_name=collection_name, points=points, wait=True)
    count = len(points)
    points.clear()
    return count


def flush_milvus_rows(
    collection_name: str,
    page_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
) -> int:
    # Send the accumulated Milvus page and token rows to the server and clear the local batch.
    if not page_rows:
        return 0

    client = get_milvus_client()
    page_collection, token_collection = milvus_collection_names(collection_name)
    ensure_milvus_collection_loaded(page_collection)
    ensure_milvus_collection_loaded(token_collection)

    client.insert(collection_name=page_collection, data=page_rows)
    if token_rows:
        client.insert(collection_name=token_collection, data=token_rows)

    count = len(page_rows)
    page_rows.clear()
    token_rows.clear()
    return count


def flush_index_batch(collection_name: str, batch: PendingIndexBatch, backend: str) -> int:
    # Flush the backend-specific batch and return the number of indexed pages committed.
    if backend == "qdrant":
        return flush_qdrant_points(collection_name, batch.qdrant_points)
    return flush_milvus_rows(collection_name, batch.milvus_pages, batch.milvus_tokens)


def qdrant_search_points(
    collection_name: str,
    query_vectors: list[list[float]],
    limit: int,
) -> list[SearchHit]:
    # Execute a Qdrant multivector search and normalize the response into search hits.
    client = get_qdrant_client()

    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=query_vectors,
            limit=limit,
            with_payload=True,
        )
        results = list(getattr(response, "points", response))
    else:
        results = list(
            client.search(
                collection_name=collection_name,
                query_vector=query_vectors,
                limit=limit,
                with_payload=True,
            )
        )

    return [
        SearchHit(
            point_id=str(hit.id),
            score=float(hit.score),
            payload=hit.payload or {},
        )
        for hit in results
    ]


def milvus_page_candidates(
    collection_name: str,
    query_vectors: list[list[float]],
    limit: int,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    # Search the Milvus page collection to build the candidate set for reranking.
    client = get_milvus_client()
    page_collection, _token_collection = milvus_collection_names(collection_name)
    ensure_milvus_collection_loaded(page_collection)

    candidate_limit = max(limit * 10, get_settings().milvus_candidate_limit)
    pooled_query = mean_pool_vectors(query_vectors)
    results = client.search(
        collection_name=page_collection,
        data=[pooled_query],
        anns_field=MILVUS_PAGE_VECTOR_FIELD,
        limit=candidate_limit,
        output_fields=["payload_json"],
        search_params={"metric_type": "COSINE"},
    )

    rows = results[0] if results else []
    candidate_ids: list[str] = []
    candidate_payloads: dict[str, dict[str, Any]] = {}
    for row in rows:
        page_id = str(row.get("id", ""))
        entity = row.get("entity") or {}
        if not page_id:
            continue
        candidate_ids.append(page_id)
        candidate_payloads[page_id] = deserialize_payload(entity.get("payload_json"))
    return candidate_ids, candidate_payloads


def milvus_search_points(
    collection_name: str,
    query_vectors: list[list[float]],
    limit: int,
) -> list[SearchHit]:
    # Execute the Milvus two-stage retrieval flow and normalize the response into search hits.
    candidate_ids, candidate_payloads = milvus_page_candidates(collection_name, query_vectors, limit)
    if not candidate_ids:
        return []

    client = get_milvus_client()
    _page_collection, token_collection = milvus_collection_names(collection_name)
    ensure_milvus_collection_loaded(token_collection)

    filter_expression = (
        f'{MILVUS_TOKEN_GROUP_FIELD} in [{", ".join(milvus_quote_string(page_id) for page_id in candidate_ids)}]'
    )

    aggregated_scores = {page_id: 0.0 for page_id in candidate_ids}
    for query_vector in query_vectors:
        grouped_results = client.search(
            collection_name=token_collection,
            data=[query_vector],
            anns_field=MILVUS_TOKEN_VECTOR_FIELD,
            filter=filter_expression,
            limit=min(len(candidate_ids), 16384),
            output_fields=[MILVUS_TOKEN_GROUP_FIELD],
            search_params={"metric_type": "COSINE"},
            group_by_field=MILVUS_TOKEN_GROUP_FIELD,
        )

        for row in (grouped_results[0] if grouped_results else []):
            entity = row.get("entity") or {}
            page_id = str(entity.get(MILVUS_TOKEN_GROUP_FIELD, ""))
            if not page_id:
                continue
            aggregated_scores[page_id] = aggregated_scores.get(page_id, 0.0) + float(
                row.get("distance", 0.0)
            )

    ranked_page_ids = sorted(
        aggregated_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]

    return [
        SearchHit(
            point_id=page_id,
            score=float(score),
            payload=candidate_payloads.get(page_id, {}),
        )
        for page_id, score in ranked_page_ids
    ]


def search_points(
    collection_name: str,
    query_vectors: list[list[float]],
    limit: int,
) -> list[SearchHit]:
    # Dispatch the search request to the configured backend implementation.
    if get_vector_db_backend() == "qdrant":
        return qdrant_search_points(collection_name, query_vectors, limit)
    return milvus_search_points(collection_name, query_vectors, limit)


def check_vector_db_health() -> str:
    # Probe the configured vector backend and return a compact status string.
    try:
        if get_vector_db_backend() == "qdrant":
            get_qdrant_client().get_collections()
        else:
            get_milvus_client().list_collections()
    except Exception as exc:
        return f"error: {exc.__class__.__name__}"
    return "ok"

# RAG_DB_Milvus

Déploiement dédié de Milvus en mode standalone pour la base vectorielle.

## Contenu

- `docker-compose.yml` : démarrage de Milvus et de ses dépendances
- `.env.example` : configuration des ports, des volumes et des noms de conteneurs
- `volumes/` : volumes locaux persistants pour `etcd`, `minio` et `milvus`

## Variables importantes

- `MILVUS_PORT` : port Milvus exposé pour les clients
- `MILVUS_HEALTH_PORT` : port HTTP de healthcheck et de Web UI Milvus
- `MILVUS_MINIO_API_PORT` : port API MinIO
- `MILVUS_MINIO_CONSOLE_PORT` : port console MinIO
- `MILVUS_STORAGE_PATH` : chemin local du stockage Milvus
- `MILVUS_ETCD_PATH` : chemin local du stockage etcd
- `MILVUS_MINIO_PATH` : chemin local du stockage MinIO
- `MILVUS_VERSION` : version de l'image Milvus
- `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` : credentials MinIO

## Démarrage

```bash
docker network create rag-network
cd RAG_DB_Milvus
cp .env.example .env
docker compose up -d
```

## Accès

- Milvus : `localhost:${MILVUS_PORT}` pour les clients
- Health / Web UI : `http://localhost:${MILVUS_HEALTH_PORT}/healthz`
- MinIO API : `http://localhost:${MILVUS_MINIO_API_PORT}`
- MinIO Console : `http://localhost:${MILVUS_MINIO_CONSOLE_PORT}`

## Exemples Python

Pour tester Milvus directement en Python :

```bash
python3 -m pip install pymilvus==2.6.11 numpy
```

### Stocker des vecteurs de page

Milvus ne gère pas nativement le multi-vecteur `MAX_SIM` comme Qdrant. Dans ce repo, on utilise donc deux niveaux :

- une collection de pages pour récupérer des candidats rapidement
- une collection de tokens pour faire le reranking late-interaction ensuite

Exemple minimal pour stocker des vecteurs de page :

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530")

collection_name = "colqwen_pages"
vector_size = 4

if collection_name in client.list_collections():
    client.drop_collection(collection_name=collection_name)

schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field(field_name="page_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_size)
schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=256)
schema.add_field(field_name="page_number", datatype=DataType.INT64)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="page_vector",
    index_type="AUTOINDEX",
    metric_type="COSINE",
)

client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params,
)

client.insert(
    collection_name=collection_name,
    data=[
        {
            "id": "page-1",
            "page_vector": [0.14, 0.42, -0.04, 0.90],
            "filename": "manuel.pdf",
            "page_number": 1,
        }
    ],
)
```

### Rechercher des pages candidates

```python
results = client.search(
    collection_name=collection_name,
    data=[[0.11, 0.40, -0.03, 0.92]],
    anns_field="page_vector",
    limit=5,
    output_fields=["filename", "page_number"],
    search_params={"metric_type": "COSINE"},
)

for hit in results[0]:
    print(hit["id"], hit["distance"], hit["entity"])
```

### Stocker les vecteurs token-level pour le reranking

Pour approcher `MAX_SIM`, vous pouvez stocker un vecteur par token avec un `page_id` commun :

```python
token_collection = "colqwen_tokens"

if token_collection in client.list_collections():
    client.drop_collection(collection_name=token_collection)

schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field(field_name="page_id", datatype=DataType.VARCHAR, max_length=64)
schema.add_field(field_name="token_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_size)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="token_vector",
    index_type="AUTOINDEX",
    metric_type="COSINE",
)

client.create_collection(
    collection_name=token_collection,
    schema=schema,
    index_params=index_params,
)

client.insert(
    collection_name=token_collection,
    data=[
        {"id": "tok-1", "page_id": "page-1", "token_vector": [0.12, 0.44, -0.05, 0.91]},
        {"id": "tok-2", "page_id": "page-1", "token_vector": [0.18, 0.39, -0.07, 0.88]},
    ],
)
```

### Calcul `MaxSim` en Python pour un candidat Milvus

Une fois les tokens d'une page candidate récupérés, le `MaxSim` se fait côté application :

```python
import numpy as np

query_vectors = [
    [0.10, 0.41, -0.02, 0.93],
    [0.16, 0.37, -0.08, 0.86],
]

doc_vectors = [
    [0.12, 0.44, -0.05, 0.91],
    [0.18, 0.39, -0.07, 0.88],
]

def maxsim(query_vectors: list[list[float]], doc_vectors: list[list[float]]) -> float:
    q = np.asarray(query_vectors, dtype=np.float32)
    d = np.asarray(doc_vectors, dtype=np.float32)
    similarities = q @ d.T
    return float(similarities.max(axis=1).sum())

score = maxsim(query_vectors, doc_vectors)
print(score)
```

Dans `RAG_Orch`, c'est cette idée qui est utilisée avec Milvus : recherche de candidats sur les pages, puis reranking token-level.

## Remarques

Ce module suit la structure de `RAG_DB`, mais Milvus nécessite aussi `etcd` et `minio`.

`RAG_Orch` peut maintenant l'utiliser directement avec une configuration du type :

```env
VECTOR_DB_BACKEND=milvus
MILVUS_URL=http://rag-db-milvus:19530
```

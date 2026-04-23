# RAG_DB

Déploiement dédié de Qdrant pour la base vectorielle.

## Contenu

- `docker-compose.yml` : démarrage du service Qdrant
- `.env.example` : configuration des ports et du stockage
- `storage/` : volume local par défaut

## Variables importantes

- `QDRANT_HTTP_PORT` : port HTTP exposé pour les clients
- `QDRANT_GRPC_PORT` : port gRPC exposé
- `QDRANT_STORAGE_PATH` : chemin local du stockage persistant
- `QDRANT_CONTAINER_NAME` : nom du container

## Démarrage

```bash
cd RAG_DB
cp .env.example .env
docker compose up -d
```

## Accès

- HTTP : `http://<host>:${QDRANT_HTTP_PORT}`
- gRPC : `<host>:${QDRANT_GRPC_PORT}`

## Exemples Python

Pour tester Qdrant directement en Python :

```bash
python3 -m pip install qdrant-client numpy
```

### Stocker des multi-vecteurs ColQwen

Qdrant sait stocker nativement un multi-vecteur par point. C'est le cas le plus simple si vous voulez garder un comportement de type late-interaction avec `MAX_SIM`.

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

collection_name = "colqwen_pages"
vector_size = 128

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM,
        ),
    ),
)

page_vectors = [
    [0.12, 0.44, -0.05, 0.91],
    [0.18, 0.39, -0.07, 0.88],
]

client.upsert(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id="page-1",
            vector=page_vectors,
            payload={
                "filename": "manuel.pdf",
                "page_number": 1,
                "chunk_text": "Exemple de page indexee",
            },
        )
    ],
    wait=True,
)
```

### Rechercher avec `MAX_SIM`

Avec un multi-vecteur de query, le score est calculé dans Qdrant. En pratique, chaque vecteur de la query prend la meilleure similarité contre les vecteurs du document, puis ces meilleurs scores sont additionnés.

```python
query_vectors = [
    [0.10, 0.41, -0.02, 0.93],
    [0.16, 0.37, -0.08, 0.86],
]

response = client.query_points(
    collection_name=collection_name,
    query=query_vectors,
    limit=5,
    with_payload=True,
)

for hit in response.points:
    print(hit.id, hit.score, hit.payload)
```

### Calcul `MaxSim` en Python pur

Si vous voulez vérifier le score côté application :

```python
import numpy as np

def maxsim(query_vectors: list[list[float]], doc_vectors: list[list[float]]) -> float:
    q = np.asarray(query_vectors, dtype=np.float32)
    d = np.asarray(doc_vectors, dtype=np.float32)
    similarities = q @ d.T
    return float(similarities.max(axis=1).sum())

score = maxsim(query_vectors, page_vectors)
print(score)
```

Ce calcul correspond à l'idée de `MultiVectorComparator.MAX_SIM` utilisée par Qdrant.

## Remarques

Ce module peut être déployé seul sur un serveur distinct. L'orchestrateur devra simplement connaître l'URL HTTP Qdrant, par exemple `http://10.0.0.20:6333`.

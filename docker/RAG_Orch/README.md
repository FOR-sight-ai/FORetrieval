# RAG_Orch

Service d'orchestration du pipeline RAG. Il ne porte pas le modèle et ne porte pas la base : il appelle `RAG_Model` pour les embeddings et un backend vectoriel externe pour l'indexation et le retrieval.

## Contenu

- `Dockerfile`
- `docker-compose.yml`
- `.env.example`
- `requirements.txt`
- `app/main.py`

## Endpoints

- `GET /health`
- `POST /v1/index`
- `POST /v1/retrieve`

## Variables importantes

- `ORCH_HOST_PORT` : port HTTP exposé
- `EMBEDDING_SERVICE_URL` : URL du service `RAG_Model`
- `VECTOR_DB_BACKEND` : backend vectoriel à utiliser, `milvus` par défaut, `qdrant` en option
- `MILVUS_URL` : URL du service Milvus
- `MILVUS_CANDIDATE_LIMIT` : nombre de pages candidates retenues avant reranking late-interaction
- `VECTOR_DB_UPSERT_BATCH_SIZE` : taille des batchs d'upsert côté backend vectoriel
- `QDRANT_URL` : URL HTTP du service Qdrant si `VECTOR_DB_BACKEND=qdrant`
- `DEFAULT_PDF_DPI` : DPI de rasterisation

## Backends pris en charge

`RAG_Orch` supporte maintenant deux backends :

- `milvus` : backend par défaut
- `qdrant` : backend conservé pour compatibilité

Avec Milvus, l'orchestrateur crée deux collections techniques par collection logique :

- une collection `__pages` pour les vecteurs agrégés par page
- une collection `__tokens` pour les vecteurs token-level utilisés au reranking

Le retrieval Milvus fonctionne en deux temps :

1. récupération d'un ensemble de pages candidates via un vecteur de page agrégé
2. reranking late-interaction sur les vecteurs token-level pour approcher le comportement multivector précédemment assuré par Qdrant

## Démarrage

### Qdrant
```bash
cd RAG_Orch
cp .env.qdrant .env
docker compose up --build
```

### Milvus
```bash
cd RAG_Orch
cp .env.milvus .env
docker compose up --build
```

## Exemple de test rapide

```bash
curl http://localhost:8000/health
```

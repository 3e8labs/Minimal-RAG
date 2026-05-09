# RAG in Pure C — Project Plan

A minimal Retrieval-Augmented Generation (RAG) pipeline written in pure C,
targeting Apple M1. No C++, no Python at runtime, no heavy frameworks.

---

## Goals

- Pure C, minimal dependencies
- Runs fully local on Apple M1
- Clean, readable antirez-style code
- Four progressively capable versions sharing the same core

---

## Models

| Role | Model | Size |
|---|---|---|
| Embeddings | GTE-Small (converted to `.gtemodel`) | ~127 MB |
| Generation | TinyLlama-1.1B-Chat Q4_K_M GGUF | ~670 MB |

### One-time model setup

```bash
# Download GTE-Small from HuggingFace
pip install huggingface_hub safetensors
python -c "from huggingface_hub import snapshot_download; \
           snapshot_download('thenlper/gte-small', local_dir='./gte-small-hf')"

# Convert to .gtemodel (only needs Python once)
python convert_model.py ./gte-small-hf gte-small.gtemodel

# Download TinyLlama GGUF
brew install llama.cpp
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir ./models

# Run llama.cpp server (keep running in background)
llama-server -m ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --port 8080
```

---

## Shared Libraries (all versions)

| File | Source | Purpose |
|---|---|---|
| `gte.c / gte.h` | github.com/antirez/gte-pure-C | Text embeddings (384-dim) |
| `sds.c / sds.h` | github.com/antirez/sds | Dynamic string handling |
| `cJSON.h` | github.com/DaveGamble/cJSON | JSON parsing (single header) |
| `libcurl` | system (macOS) | HTTP calls to llama.cpp server |

---

## Shared Source Files (yours)

| File | Purpose |
|---|---|
| `main.c` | CLI: `./rag ingest <file>` and `./rag query "<text>"` |
| `chunk.c / chunk.h` | Split documents into text chunks |
| `embed.c / embed.h` | Generate embeddings via gte-pure-C |
| `llm.c / llm.h` | Call llama.cpp server via libcurl, build prompt with context |

---

## Architecture

```
INGEST:
[Text file] → chunk.c → embed.c (gte-pure-C) → store (version-specific)

QUERY:
[Query text] → embed.c → retrieve top-k chunks → llm.c → answer
```

---

## Version 1 — Flat Binary File

**Storage:** Custom flat binary file on disk
**Retrieval:** Brute-force linear scan with cosine similarity

### Extra files
None — fully self-contained.

### Binary format
```
[uint32 num_chunks]
[chunk_0: uint32 text_len | char text[] | float embedding[384]]
[chunk_1: ...]
...
```

### Retrieval
```c
// Load all chunks into memory
// For each chunk: score = gte_cosine_similarity(query_emb, chunk_emb, 384)
// Return top-k by score
```

### Compile
```bash
cc -O3 -march=native -ffast-math \
   main.c gte.c sds.c chunk.c embed.c store_v1.c llm.c \
   -lm -lcurl -o rag
```

### When to use
- Small document sets (< 5000 chunks)
- No dependencies whatsoever
- Learning / prototyping

---

## Version 2 — In-Memory Hash Table (antirez dict)

**Storage:** antirez's `dict.c` hash table (from Redis)
**Retrieval:** Linear scan over hash table values

### Extra files
`dict.c / dict.h` — from github.com/antirez/historical-redis-versions

### Structure
- Key: chunk ID (integer as string)
- Value: struct containing text (`sds`) + embedding (`float[384]`)

### When to use
- Need fast lookup by chunk ID
- Want to stay in the antirez ecosystem
- Still comfortable with linear scan at retrieval time

---

## Version 3 — sqlite-vec

**Storage:** SQLite database + sqlite-vec extension
**Retrieval:** Proper KNN with HNSW approximate nearest neighbour index

### Extra files
`sqlite-vec.c` — github.com/asg017/sqlite-vec (single file drop-in)

### Schema
```sql
CREATE VIRTUAL TABLE chunks USING vec0(
    embedding FLOAT[384]
);
CREATE TABLE chunk_text (id INTEGER PRIMARY KEY, text TEXT);
```

### Retrieval
```sql
SELECT chunk_text.text, distance
FROM chunks
JOIN chunk_text ON chunks.rowid = chunk_text.id
WHERE embedding MATCH ? AND k = 5
ORDER BY distance;
```

### When to use
- Larger document sets (10k+ chunks)
- Need persistent, queryable index
- Production-grade retrieval quality

---

## Version 4 — Custom Vector Index

**Storage:** Your own pure C vector index implementation
**Retrieval:** Custom HNSW or IVF approximate KNN, zero external deps

### Approach
- Implement a simple flat IVF (inverted file index) or basic HNSW graph in C
- Store index in a binary file with a custom format
- No sqlite, no external libs

### When to use
- Full control over the index
- Deep understanding of ANN search
- Truly zero dependencies

---

## Final Directory Structure

```
rag_c/
├── Plan.md
├── Decisions.md
├── Makefile
│
├── gte.c / gte.h              # antirez — embeddings
├── sds.c / sds.h              # antirez — strings
├── dict.c / dict.h            # antirez — hash table (v2)
├── cJSON.h                    # DaveGamble — JSON
├── sqlite-vec.c               # asg017 — vector search (v3)
│
├── main.c                     # CLI entry point
├── chunk.c / chunk.h          # text chunking
├── embed.c / embed.h          # embedding wrapper
├── llm.c / llm.h              # LLM HTTP calls
│
├── store_v1.c / store_v1.h    # flat binary file
├── store_v2.c / store_v2.h    # dict hash table
├── store_v3.c / store_v3.h    # sqlite-vec
├── store_v4.c / store_v4.h    # custom vector index
│
└── models/
    ├── gte-small.gtemodel
    └── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

---

## Compile flags for M1

```bash
# Without BLAS (pure C, no frameworks)
cc -O3 -march=native -ffast-math ...

# With Apple Accelerate (faster matrix ops in gte.c)
cc -O3 -march=native -ffast-math -DUSE_BLAS -framework Accelerate ...
```

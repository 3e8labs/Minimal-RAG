# RAG in Pure C

A minimal Retrieval-Augmented Generation (RAG) pipeline written in **pure C**, targeting Apple M1. No C++, no Python at runtime, no heavy frameworks.

## Overview

This project implements a complete RAG system that:
1. **Chunks** documents into overlapping text segments
2. **Embeds** chunks using GTE-Small (384-dimensional embeddings)
3. **Stores** and **retrieves** chunks with configurable backends (flat file, hash table, SQLite, or custom index)
4. **Generates** responses by calling a local LLM server with retrieved context

The codebase is designed for learning—every component is implemented from first principles with readable, antirez-style C code.

## Quick Start

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3, etc.)
- **Xcode Command Line Tools**: `xcode-select --install`
- **libcurl** (usually pre-installed on macOS)
- **llama.cpp** for the LLM server: `brew install llama.cpp`

### 1. Download & Convert Models

```bash
# Download GTE-Small embedding model
pip install huggingface_hub safetensors
python -c "from huggingface_hub import snapshot_download; \
           snapshot_download('thenlper/gte-small', local_dir='./models/gte-small-hf')"

# Convert to .gtemodel format (included in the repo)
python convert_model.py ./models/gte-small-hf ./models/gte-small.gtemodel

# Download TinyLlama LLM (optional, for generation)
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir ./models
```

### 2. Start the LLM Server (in a separate terminal)

```bash
llama-server -m ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --port 8080
```

### 3. Build & Run

```bash
# Build the executable
cc -O3 -march=native -ffast-math \
   main.c gte.c sds.c chunk.c embed.c store_v1.c llm.c \
   -lm -lcurl -o rag

# Ingest a document
./rag --model ./models/gte-small.gtemodel --file sample.txt

# Query the document
./rag --model ./models/gte-small.gtemodel --file sample.txt --query "What is this about?" --k 3
```

## Usage

### Ingestion

```bash
./rag --model <path.gtemodel> --file <document.txt>
```

- Reads the document
- Splits it into overlapping chunks (1000 chars, 200 char overlap by default)
- Embeds each chunk using GTE-Small
- Stores chunks and embeddings in `store_v1.dat`

### Querying

```bash
./rag --model <path.gtemodel> --file <document.txt> --query "your question here" --k 5
```

- Embeds the query
- Retrieves top-k most similar chunks (by cosine similarity)
- Prints the results with scores

## Project Structure

```
rag_c/
├── README.md
├── Plan.md                    # Architecture and version roadmap
├── CLAUDE.md                  # Learning principles
│
├── Core Dependencies (3rd-party)
├── gte.c / gte.h              # GTE embedding model (antirez)
├── sds.c / sds.h              # Dynamic strings (antirez)
├── cJSON.h                    # JSON parsing (DaveGamble)
├── third_party/gte/           # GTE model source
│
├── Shared Source (you write)
├── main.c                     # CLI entry point
├── chunk.c / chunk.h          # Document chunking
├── embed.c / embed.h          # Embedding wrapper
├── llm.c / llm.h              # LLM HTTP calls
│
├── Storage Backends
├── store_v1.c / store_v1.h    # Flat binary file (ACTIVE)
├── store_v2.c / store_v2.h    # Hash table (antirez dict)
├── store_v3.c / store_v3.h    # SQLite + vector index
├── store_v4.c / store_v4.h    # Custom vector index (planned)
│
└── models/
    ├── gte-small.gtemodel     # Embedding model
    └── tinyllama-*.gguf       # LLM model
```

## Versions

The project supports four progressively advanced storage/retrieval backends:

| Version | Storage | Retrieval | When to Use |
|---------|---------|-----------|------------|
| **v1** | Flat binary file | Linear scan + cosine similarity | Learning, small datasets (<5k chunks) |
| **v2** | Hash table (dict) | Linear scan with fast ID lookup | Want Redis-style data structures |
| **v3** | SQLite + sqlite-vec | HNSW approximate KNN | Production, larger datasets (10k+ chunks) |
| **v4** | Custom C index | Pure C HNSW or IVF | Full control, zero dependencies |

Currently **v1 is implemented**. See `Plan.md` for roadmap.

## How It Works

### Embeddings

The project uses **GTE-Small**, a 127MB embedding model that produces 384-dimensional vectors. Since GTE embeddings are L2-normalized, the dot product between two embeddings equals their cosine similarity—no special normalization needed.

```c
similarity = dot_product(query_embedding, chunk_embedding, 384);
```

### Chunking

Documents are split into overlapping windows:
- **Chunk size**: 1000 characters
- **Overlap**: 200 characters
- Simple byte-based splitting (no fancy tokenization)

### Storage (v1)

A custom binary format stores chunks and embeddings efficiently:

```
[uint32 num_chunks]
[chunk_0: uint32 text_len | char text[] | float embedding[384]]
[chunk_1: ...]
```

### Retrieval

Brute-force linear scan over all chunks, scoring each with cosine similarity:

```c
for (each chunk) {
    score = dot_product(query_embedding, chunk_embedding, 384)
    if (score in top-k) insert into results
}
return top-k sorted by score
```

## Compilation Flags

### Standard (Pure C, no frameworks)
```bash
cc -O3 -march=native -ffast-math ...
```

### With Apple Accelerate (faster matrix ops)
```bash
cc -O3 -march=native -ffast-math -DUSE_BLAS -framework Accelerate ...
```

## Code Style

The codebase follows **antirez principles**:
- Readable, explicit C with minimal macros
- No over-engineering or premature abstraction
- Short, focused functions with clear error handling
- Comments explaining *why*, not what

## Learning Resources

- **CLAUDE.md**: Philosophy and rules for this learning project
- **Plan.md**: Detailed architecture and future versions
- **Decisions.md**: Design decisions and tradeoffs
- **dev_log.md**: Development progress and notes

## Contributing

This is a learning project. Each component is designed to teach one concept deeply:
- How embeddings work
- Binary file formats
- Cosine similarity
- C memory management
- HTTP communication

See `CLAUDE.md` for the project's learning principles.

## License

This project includes code from:
- **antirez**: gte.c, sds.c (original Redis/single-purpose implementations)
- **DaveGamble**: cJSON
- **asg017**: sqlite-vec (for v3)

See individual files for license details.

## Status

- ✅ **v1**: Flat binary file storage (complete)
- 🔄 **v2-v4**: In progress (see Plan.md)

## Questions?

Run `./rag` with no arguments for usage help.

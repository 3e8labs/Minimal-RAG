# Design Decisions — RAG in Pure C

A record of why each technology was chosen or rejected during planning.

---

## Language: Pure C

**Decision:** Pure C only. No C++, no Python at runtime.

**Why:** The goal is a minimal, dependency-free, portable pipeline. C gives full
control over memory, has no runtime overhead, and forces simplicity. The
antirez aesthetic — small, readable, self-contained — is only achievable in C.

---

## Why not Python?

Python is the obvious choice for RAG (langchain, llama-index, sentence-transformers
all exist). Rejected because:
- Too many dependencies (pip, venv, packages)
- Not portable to embedded or constrained environments
- Goes against the minimal, pure C goal

---

## Why not ds4 (antirez)?

**Repo:** github.com/antirez/ds4

Considered because antirez wrote it, it is pure C, and it targets M1 Metal.

**Rejected because:**
- Hardcoded to DeepSeek V4 Flash only — a large MoE model, not minimal
- No embedding API exposed in `ds4.h` — hidden states exist internally
  (`comp_pooled`, `attn_residual`) but are not accessible from outside
- DeepSeek V4 Flash is ~12-14 GB in Q4, violates the "smallest possible model" goal

---

## Why not gguf-tools (antirez)?

**Repo:** github.com/antirez/gguf-tools

A pure C library for reading and writing GGUF files.

**Not needed because:**
- `gte-pure-C` uses its own `.gtemodel` format, not GGUF
- `llama.cpp` server handles GGUF loading internally
- Our C code never touches a GGUF file directly

Useful for inspecting model files (`gguf-tools show model.gguf`) but not
a runtime dependency.

---

## Embedding model: GTE-Small

**Alternatives considered:** all-MiniLM-L6-v2, Qwen3-ASR-1.7B

**Chose GTE-Small because:**
- Higher MTEB retrieval benchmark scores than all-MiniLM at similar size
- 33M parameters, 384-dim embeddings — small and fast
- Official GGUF from ggml-org available
- Pure C implementation exists: `gte-pure-C` by antirez

**Rejected all-MiniLM:** GTE-Small benchmarks better at similar size.

**Rejected Qwen3-ASR-1.7B:** ASR = Automatic Speech Recognition. It converts
audio to text, not a text embedding or generation model. Wrong tool entirely.

---

## Embedding library: gte-pure-C (antirez)

**Repo:** github.com/antirez/gte-pure-C

**Chose because:**
- Pure C, zero dependencies
- ~900 lines total
- Implements full BERT transformer: WordPiece tokenizer, self-attention,
  FFN, mean pooling, L2 normalisation
- Optional Apple Accelerate support (`-DUSE_BLAS`) for M1 speedup
- Output matches PyTorch numerically
- Clean 6-function API: `gte_load`, `gte_embed`, `gte_cosine_similarity`, etc.
- Written by antirez — same style as rest of the project

**Known caveats:**
- README states "no accurate testing performed for all tokenizer edge cases"
- Tokenizer is ASCII-only — non-ASCII Unicode falls back to `[UNK]`
- Not thread-safe — shared working buffers in context struct
- Model stored as unquantized float32 (~127 MB, not 33 MB like quantized GGUF)
- Requires one-time Python conversion (`convert_model.py`) to produce `.gtemodel`

**Assessment:** Caveats are edge cases that do not affect English-language RAG.
The math (layer norm, attention, GELU, softmax, mean pool) is correct.

---

## Generation model: TinyLlama-1.1B-Chat

**Alternatives considered:** SmolLM2-1.7B, Qwen3-1.7B, DeepSeek V4 Flash

**Chose TinyLlama-1.1B because:**
- Smallest viable chat-capable model: 1.1B parameters
- Q4_K_M GGUF: ~670 MB
- Trained on 3 trillion tokens, good quality for its size
- Runs well on M1 via llama.cpp with Metal

**Rejected DeepSeek V4 Flash:** Large MoE model (~12-14 GB). ds4 is the only
pure C runner for it, and ds4 exposes no embedding API.

---

## Generation runtime: llama.cpp server

**Why server mode, not direct linking:**
- llama.cpp is C++, not C — linking against it would break the pure C goal
- Server mode exposes an OpenAI-compatible HTTP API
- Our C code calls it via libcurl — clean separation
- Metal acceleration on M1 handled by llama.cpp internally

---

## String library: sds (antirez)

**Repo:** github.com/antirez/sds

**Chose because:**
- Pure C, zero dependencies, two files (`sds.c`, `sds.h`)
- Battle-tested — used in Redis since 2009, still maintained
- Useful for: building HTTP request bodies, accumulating chunked text,
  `sdssplitlen` for document splitting, `sdscatprintf` for prompt building
- antirez-style, fits the project aesthetic

**Found via:** antirez's historical-redis-versions repo (2009 Redis source),
then located the modern standalone version.

---

## JSON parsing: cJSON

**Chose because:**
- Single header file, pure C, zero dependencies
- Sufficient for parsing llama.cpp API responses
- Widely used, well tested

---

## HTTP: libcurl

**Chose because:**
- Already present on macOS, no installation needed
- Standard, reliable C HTTP library
- Only needed for calling the llama.cpp generation server

---

## Vector storage: four versions

**Why not just one approach:**
The right storage depends on document scale and complexity goals. Rather than
pick one, the project implements four progressively capable versions that share
the same embedding, chunking, and LLM code — only the storage layer swaps.

### Version 1: Flat binary file
- Custom binary format: `[num_chunks][text_len|text|embedding[384]]...`
- Retrieval: linear scan with `gte_cosine_similarity`
- Zero dependencies, ~100 lines of C
- **Why viable:** 5000 chunks × 384 dims = ~7.7M floats = a few ms on M1

### Version 2: antirez dict.c
- In-memory hash table from historical Redis source
- Key: chunk ID, Value: text + embedding
- Linear scan for similarity, O(1) lookup by ID
- Stays fully in the antirez ecosystem

### Version 3: sqlite-vec
- SQLite + sqlite-vec extension (single `.c` file by asg017)
- HNSW approximate KNN index
- Best for 10k+ chunks, persistent index, SQL queryable
- sqlite3 is already on macOS

### Version 4: Custom vector index
- Implement HNSW or IVF from scratch in pure C
- Truly zero dependencies
- Educational — understand ANN search deeply

---

## Rejected: Redis / Redis Stack

The user prefers the antirez style and considered Redis as the vector store.

**Rejected because:**
- Full Redis is too heavy for a minimal local RAG
- Redis vector search requires Redis Stack 7.x with RediSearch module — large C++ codebase
- Requires a running Redis server (external dependency)
- No minimal Redis with vector search exists anywhere on GitHub (searched thoroughly)
- sqlite-vec matches the minimalism goal better

**Antirez's other data structure repos** (`rax`, `listpack`, `otree`) are
interesting but none provide vector similarity search.

---

## Rejected: Qwen3-ASR

User asked about Qwen3-ASR-1.7B.

**Rejected because:** ASR = Automatic Speech Recognition. Converts audio to text.
Not a text LLM or embedding model. Entirely wrong category for RAG.

The correct Qwen3 model for generation would be Qwen3-1.7B (text LLM),
but TinyLlama was already chosen for its smaller size.

---

## Platform: Apple M1

All decisions assume M1 Mac:
- `gte.c` uses `-DUSE_BLAS` with `Accelerate.framework` for BLAS acceleration
- `llama.cpp` uses Metal GPU backend for fast inference
- Compile with `-O3 -march=native -ffast-math` for NEON vectorisation
- `sqlite3` and `libcurl` are pre-installed on macOS

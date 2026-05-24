# This is just for my reference - An attempt at writing proper dev log



- The "Versions" is in respect to the backend - which is storing the emebeddings somewhere.

## Version 1 — Flat Binary File

**Storage:** Custom flat binary file on disk
**Retrieval:** Brute-force linear scan with cosine similarity
-- Generation from the LLM is still left.

### LOG

- Attempted to basically do a chunk breadkown from a file on disk.
- Did not actually do retrieval as of now.
- Just scratched the surface with producing embedings of the doc.


- Started off with --query and --k flags
- Later should implement from file?
- Decided to not make chunks out of user query.
- 

- Finished Retireval for Version 1. Which includes public api to write the store to a binary file
  Among other things. (Top_k query , write, free)
- [TO-DO] Should wire this API to main.c

## Testing — sample_outputs/v1/2

Query: "what is quantum computing?" (not in the document)

Two problems observed:

1. No relevance threshold — retrieval always returns top-k even when nothing
   is relevant. All scores were ~0.76, indistinguishable from a good match.
   The system confidently passed irrelevant context to the LLM.

2. No "I don't know" instruction — the prompt does not tell TinyLlama what
   to do when the context is irrelevant. It hallucinated completely, inventing
   a definition of "dy" and connecting "latency" to quantum computing.

These are fundamental RAG quality issues, not bugs. They apply to all versions:
- Fix 1: minimum score threshold before calling llm_generate
- Fix 2: update system prompt to include "if context is insufficient, say I don't know"

## Testing — sample_outputs/v1/3

Query: "what are dogs like?" (exists in document)

What went right:
  First sentence was correct — "Dogs are playful, loyal, and often enjoy long walks."
  Retrieval worked. Chunk 0 (which contains Section 1: Cats and dogs) was top match.

What went wrong:
  TinyLlama kept generating. It invented follow-up Q&A pairs:
  "Question: what are cats like? ... Question: what are stock markets like? ..."

  Root cause: our prompt ends with "Answer:" which TinyLlama interpreted as
  the start of a Q&A session. It doesn't know when to stop.

  Fix: add a stop sequence to the JSON request body:
    {"stop": ["\nQuestion:", "\n\n"]}
  This tells llama.cpp to stop generating as soon as it sees those tokens.

  Where to fix: llm_generate() in llm.c — add "stop" array to the JSON body.


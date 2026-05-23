#ifndef LLM_H
#define LLM_H

typedef struct llm_ctx llm_ctx;

/* Load an LLM context pointing at a llama.cpp server URL.
 * Example: llm_load("http://localhost:8080")
 * Returns NULL on error. */
llm_ctx *llm_load(const char *server_url);

/* Free the LLM context. */
void llm_free(llm_ctx *ctx);

/* Generate an answer given a query and a context string.
 * Context is the pre-built string of retrieved chunks.
 * Returns a heap-allocated NUL-terminated string. Caller must free().
 * Returns NULL on error. */
char *llm_generate(llm_ctx *ctx, const char *query, const char *context);

#endif

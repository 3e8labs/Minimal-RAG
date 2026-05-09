#ifndef EMBED_H
#define EMBED_H

/* Embedding wrapper around vendored gte-pure-C. */

typedef struct embed_ctx embed_ctx;

/* Load a .gtemodel file. Returns NULL on error. */
embed_ctx *embed_load(const char *model_path);

/* Free the embedding context. */
void embed_free(embed_ctx *ctx);

/*
 * Embed multiple texts.
 * Returns a newly allocated float array of size: count * embed_dim(ctx).
 * Embeddings are stored contiguously: [emb0, emb1, ...].
 * Caller must free() the returned pointer.
 * Returns NULL on error (no partial results).
 */
float *embed_batch(embed_ctx *ctx, const char **texts, int count);

/* Embedding dimension (384 for GTE-small). Returns 0 on error. */
int embed_dim(embed_ctx *ctx);

#endif

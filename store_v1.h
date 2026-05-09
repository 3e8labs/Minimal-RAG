#ifndef STORE_V1_H
#define STORE_V1_H

/*
 * Version 1 store: flat binary file + brute-force scan.
 *
 * On-disk format (little-endian, as written by the host):
 *   [uint32 num_chunks]
 *   [uint32 dim]
 *   Repeated num_chunks times:
 *     [uint32 text_len]
 *     [uint8  text[text_len]]   (no NUL terminator)
 *     [float  embedding[dim]]   (float32)
 */

#include <stdint.h>

typedef struct {
    int index;   /* chunk index in the file (0..num_chunks-1) */
    float score; /* cosine similarity (dot product for normalized vectors) */
    char *text;  /* heap-allocated NUL-terminated chunk text */
} store_v1_hit;

/*
 * Write a store file.
 *
 * - `chunks` is an array of NUL-terminated strings.
 * - `embeddings` is a contiguous array: [emb0, emb1, ...] where each emb is `dim` floats.
 *
 * Returns 0 on success, non-zero on error.
 */
int store_v1_write(const char *path,
                   char **chunks,
                   int num_chunks,
                   const float *embeddings,
                   int dim);

/*
 * Query the store using brute-force linear scan.
 *
 * Returns up to k hits, sorted by descending score.
 * On success (return 0), `*out_hits` points to a heap-allocated array of `*out_count` hits.
 * Caller must free results with `store_v1_free_hits`.
 *
 * Returns 0 on success, non-zero on error.
 */
int store_v1_query_topk(const char *path,
                        const float *query_emb,
                        int dim,
                        int k,
                        store_v1_hit **out_hits,
                        int *out_count);

/* Free hits returned by `store_v1_query_topk`. Safe to call with NULL. */
void store_v1_free_hits(store_v1_hit *hits, int count);

#endif


#include "store_v1.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Tiny binary I/O helpers.
 *
 * Note: This store is intentionally "v1 simple": it writes host-endian
 * uint32 and IEEE float32 as-is. This is fine for local learning on one
 * machine, but would need explicit endianness handling for portability.
 * ------------------------------------------------------------------------- */

static int write_u32(FILE *fp, uint32_t v) {
    return fwrite(&v, sizeof(v), 1, fp) == 1 ? 0 : -1;
}

static int read_u32(FILE *fp, uint32_t *out) {
    return fread(out, sizeof(*out), 1, fp) == 1 ? 0 : -1;
}

static int write_f32(FILE *fp, const float *v, int n) {
    if (n <= 0) return 0;
    return fwrite(v, sizeof(float), (size_t)n, fp) == (size_t)n ? 0 : -1;
}

static int read_f32(FILE *fp, float *v, int n) {
    if (n <= 0) return 0;
    return fread(v, sizeof(float), (size_t)n, fp) == (size_t)n ? 0 : -1;
}

static float dot_product(const float *a, const float *b, int n) {
    float dot = 0.0f;
    for (int i = 0; i < n; i++) dot += a[i] * b[i];
    return dot;
}

/* -------------------------------------------------------------------------
 * Public API.
 * ------------------------------------------------------------------------- */

int store_v1_write(const char *path,
                   char **chunks,
                   int num_chunks,
                   const float *embeddings,
                   int dim) {
    if (!path || !chunks || !embeddings) return -1;
    if (num_chunks < 0 || dim <= 0) return -1;

    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;

    if (write_u32(fp, (uint32_t)num_chunks) != 0 ||
        write_u32(fp, (uint32_t)dim) != 0) {
        fclose(fp);
        return -1;
    }

    for (int i = 0; i < num_chunks; i++) {
        if (!chunks[i]) { fclose(fp); return -1; }
        size_t text_len = strlen(chunks[i]);
        if (text_len > UINT32_MAX) { fclose(fp); return -1; }

        if (write_u32(fp, (uint32_t)text_len) != 0) { fclose(fp); return -1; }
        if (text_len && fwrite(chunks[i], 1, text_len, fp) != text_len) { fclose(fp); return -1; }

        const float *emb = embeddings + (i * dim);
        if (write_f32(fp, emb, dim) != 0) { fclose(fp); return -1; }
    }

    if (fclose(fp) != 0) return -1;
    return 0;
}

int store_v1_query_topk(const char *path,
                        const float *query_emb,
                        int dim,
                        int k,
                        store_v1_hit **out_hits,
                        int *out_count) {
    if (!path || !query_emb || !out_hits || !out_count) return -1;
    if (dim <= 0 || k <= 0) return -1;

    *out_hits = NULL;
    *out_count = 0;

    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;

    uint32_t n_chunks_u32 = 0;
    uint32_t dim_u32 = 0;
    if (read_u32(fp, &n_chunks_u32) != 0 || read_u32(fp, &dim_u32) != 0) {
        fclose(fp);
        return -1;
    }
    if ((int)dim_u32 != dim) {
        fclose(fp);
        return -1;
    }

    int n_chunks = (int)n_chunks_u32;
    int kk = k;
    if (kk > n_chunks) kk = n_chunks;
    if (kk <= 0) { fclose(fp); return 0; }

    store_v1_hit *hits = calloc((size_t)kk, sizeof(*hits));
    if (!hits) { fclose(fp); return -1; }
    for (int i = 0; i < kk; i++) {
        hits[i].index = -1;
        hits[i].score = -2.0f;
        hits[i].text = NULL;
    }

    float *tmp_emb = malloc((size_t)dim * sizeof(float));
    if (!tmp_emb) {
        store_v1_free_hits(hits, kk);
        fclose(fp);
        return -1;
    }

    for (int idx = 0; idx < n_chunks; idx++) {
        uint32_t text_len_u32 = 0;
        if (read_u32(fp, &text_len_u32) != 0) goto fail;

        size_t text_len = (size_t)text_len_u32;
        char *text = malloc(text_len + 1);
        if (!text) goto fail;
        if (text_len && fread(text, 1, text_len, fp) != text_len) { free(text); goto fail; }
        text[text_len] = '\0';

        if (read_f32(fp, tmp_emb, dim) != 0) { free(text); goto fail; }

        float score = dot_product(query_emb, tmp_emb, dim);

        /* Insert into sorted hits[] (descending), shifting down. */
        for (int r = 0; r < kk; r++) {
            if (score > hits[r].score) {
                /* Shift tail down by 1 (and free the displaced text at the end). */
                if (hits[kk - 1].text) free(hits[kk - 1].text);
                for (int s = kk - 1; s > r; s--) hits[s] = hits[s - 1];

                hits[r].index = idx;
                hits[r].score = score;
                hits[r].text = text;
                text = NULL;
                break;
            }
        }

        if (text) free(text); /* Not in top-k. */
    }

    free(tmp_emb);
    fclose(fp);
    *out_hits = hits;
    *out_count = kk;
    return 0;

fail:
    free(tmp_emb);
    store_v1_free_hits(hits, kk);
    fclose(fp);
    return -1;
}

void store_v1_free_hits(store_v1_hit *hits, int count) {
    if (!hits) return;
    for (int i = 0; i < count; i++) free(hits[i].text);
    free(hits);
}


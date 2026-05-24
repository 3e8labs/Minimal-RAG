#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chunk.h"
#include "embed.h"
#include "llm.h"
#include "store_v1.h"

/* Chunking parameters for this smoke-test CLI (character/byte based). */
#define CHUNK_SIZE 1000
#define CHUNK_OVERLAP 200

typedef struct {
    const char *model_path;
    const char *file_path;
    const char *query;
    const char *server_url; /* optional: llama.cpp server e.g. http://localhost:8080 */
    const char *save_path;  /* optional: write binary store to this path */
    const char *load_path;  /* optional: load binary store from this path */
    int k; /* top-k for query mode */
} args;

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --model <path.gtemodel> --file <doc.txt> [--query <text>] [--k <int>] [--server <url>]\n"
        "       %s --model <path.gtemodel> --file <doc.txt> --save <store.bin>\n"
        "       %s --load <store.bin> --model <path.gtemodel> --query <text> [--server <url>]\n",
        prog, prog, prog);
}

/*
 * Parse a tiny flag-based CLI.
 *
 * We keep this explicit (no getopt) to stay "pure C" and readable.
 * Returns 0 on success, non-zero on error.
 */
static int parse_args(args *a, int argc, char **argv) {
    a->model_path = NULL;
    a->file_path = NULL;
    a->query = NULL;
    a->server_url = NULL;
    a->save_path = NULL;
    a->load_path = NULL;
    a->k = 3; // Default k value set to 3

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) a->model_path = argv[++i];
        else if (!strcmp(argv[i], "--file") && i + 1 < argc) a->file_path = argv[++i];
        else if (!strcmp(argv[i], "--query") && i + 1 < argc) a->query = argv[++i];
        else if (!strcmp(argv[i], "--server") && i + 1 < argc) a->server_url = argv[++i];
        else if (!strcmp(argv[i], "--save") && i + 1 < argc) a->save_path = argv[++i];
        else if (!strcmp(argv[i], "--load") && i + 1 < argc) a->load_path = argv[++i];
        else if (!strcmp(argv[i], "--k") && i + 1 < argc) a->k = atoi(argv[++i]);
        else return -1;
    }

    /* --save and --load are mutually exclusive */
    if (a->save_path && a->load_path) {
        fprintf(stderr, "error: --save and --load cannot be used together\n");
        return -1;
    }

    /* --load mode only needs model + query (no file) */
    if (a->load_path) {
        if (!a->model_path || !a->query) return -1;
        return 0;
    }
    if (!a->model_path || !a->file_path) return -1;
    if (a->k <= 0) return -1;
    return 0;
}

/*
 * Read a whole file into a single NUL-terminated buffer.
 *
 * We use fseek/ftell to determine size first, then do a single read.
 * Returns NULL on error. Caller owns the returned buffer.
 */
static char *read_entire_file(const char *path, long *len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    // Fseek to the END.
    long n = ftell(fp); 
    // n will be set to the length of the file in bytes.
    if (n < 0) { fclose(fp); return NULL; }
    if (fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return NULL; }
    // Setting back the file pointer to the START of the file

    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(fp); return NULL; }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); free(buf); return NULL; }
    // Reading the size of 1 char of n blocks from the SEEK_SET fp and saving to buf
    fclose(fp);
    buf[n] = '\0';
    if (len) *len = n;
    return buf;
}

/* Since GTE embeddings are L2-normalized, dot product == cosine similarity. */
// Normalised means they are divided by magnitude of the vectors.
// n is the dims. Which means it is the no.of dimensions in a embedding.
static float dot_product(const float *a, const float *b, int n) {
    float dot = 0.0f;
    for (int i = 0; i < n; i++) dot += a[i] * b[i];
    return dot;
}

static void print_embedding_preview(const float *emb, int chunk_index, int dim, int nprint) {
    if (!emb || dim <= 0 || chunk_index < 0) return;
    if (nprint > dim) nprint = dim;

    printf("emb[%d][0..%d):", chunk_index, nprint);
    const float *base = emb + (chunk_index * dim);
    for (int i = 0; i < nprint; i++) printf(" %.6f", base[i]);
    printf("\n");
}

/*
 * Brute-force top-k retrieval against all chunk embeddings.
 *
 * Returns 0 on success, non-zero on error. Prints results.
 */
static int retrieve_topk(embed_ctx *ectx,
                         const char *query,
                         int k,
                         const float *chunk_emb,
                         int num_chunks,
                         int dim,
                         char **chunks,
                         char **out_context) {
    if (!ectx || !query || !chunk_emb || !chunks) return -1;
    if (num_chunks <= 0 || dim <= 0) return -1;

    const char *qtexts[1] = { query };
    float *qemb = embed_batch(ectx, qtexts, 1);
    if (!qemb) return -1;

    int kk = k;
    if (kk > num_chunks) kk = num_chunks;

    int *best_idx = calloc((size_t)kk, sizeof(int));
    float *best_score = malloc((size_t)kk * sizeof(float));
    if (!best_idx || !best_score) {
        free(best_idx);
        free(best_score);
        free(qemb);
        return -1;
    }
    for (int i = 0; i < kk; i++) { best_idx[i] = -1; best_score[i] = -2.0f; }

    for (int c = 0; c < num_chunks; c++) {
        const float *ce = chunk_emb + (c * dim);
        float score = dot_product(qemb, ce, dim);

        for (int r = 0; r < kk; r++) {
            if (score > best_score[r]) {
                for (int s = kk - 1; s > r; s--) {
                    best_score[s] = best_score[s - 1];
                    best_idx[s] = best_idx[s - 1];
                }
                best_score[r] = score;
                best_idx[r] = c;
                break;
            }
        }
    }

    printf("\nTop %d matches:\n", kk);
    for (int r = 0; r < kk; r++) {
        if (best_idx[r] < 0) continue;
        printf("%d) score=%.4f chunk=%d\n%s\n\n",
               r + 1, best_score[r], best_idx[r], chunks[best_idx[r]]);
    }

    /* ----------------------------------------------------------------
     * Build context string from top-k chunks for LLM generation.
     * Two passes: first measure, then fill.
     * ---------------------------------------------------------------- */
    if (out_context) {
        /* Pass 1 — measure total size needed */
        size_t total = 1; /* for \0 */
        for (int r = 0; r < kk; r++) {
            if (best_idx[r] < 0) continue;
            total += strlen(chunks[best_idx[r]]);
            if (r < kk - 1) total += 3; /* "\n\n\n" separator */
        }

        char *context = malloc(total);
        if (context) {
            /* Pass 2 — fill */
            context[0] = '\0';
            for (int r = 0; r < kk; r++) {
                if (best_idx[r] < 0) continue;
                strcat(context, chunks[best_idx[r]]);
                if (r < kk - 1) strcat(context, "\n\n\n");
            }
        }
        *out_context = context;
    }

    free(best_idx);
    free(best_score);
    free(qemb);
    return 0;
}

/* Generate an answer using the LLM server if --server was given, and print it. */
static void generate_and_print(const char *server_url, const char *query, const char *context) {
    if (!server_url || !context) return;
    llm_ctx *lctx = llm_load(server_url);
    if (!lctx) { fprintf(stderr, "error: llm_load failed\n"); return; }
    char *answer = llm_generate(lctx, query, context);
    if (!answer) {
        fprintf(stderr, "error: llm_generate failed\n");
    } else {
        printf("\nAnswer:\n%s\n", answer);
        free(answer);
    }
    llm_free(lctx);
}

int main(int argc, char **argv) {
    /* --------------------------------------------------------------------
     * Parse flags.
     * -------------------------------------------------------------------- */
    args a;
    if (parse_args(&a, argc, argv) != 0) {
        usage(argv[0]);
        fprintf(stderr, "error: invalid arguments\n");
        return 1;
    }

    /* --------------------------------------------------------------------
     * Load embedding model (needed in all modes).
     * -------------------------------------------------------------------- */
    embed_ctx *ectx = embed_load(a.model_path);
    if (!ectx) {
        fprintf(stderr, "error: embed_load failed (model: %s)\n", a.model_path);
        return 1;
    }
    int dim = embed_dim(ectx);

    /* ====================================================================
     * MODE 2: --load + --query
     * Embed the query, scan the binary store, build context, generate.
     * ==================================================================== */
    if (a.load_path) {
        const char *qtexts[1] = { a.query };
        float *qemb = embed_batch(ectx, qtexts, 1);
        if (!qemb) {
            fprintf(stderr, "error: embed_batch failed for query\n");
            embed_free(ectx);
            return 1;
        }

        store_v1_hit *hits = NULL;
        int hit_count = 0;
        if (store_v1_query_topk(a.load_path, qemb, dim, a.k, &hits, &hit_count) != 0) {
            fprintf(stderr, "error: store_v1_query_topk failed\n");
            free(qemb);
            embed_free(ectx);
            return 1;
        }
        free(qemb);

        printf("query:  %s\nk:      %d\nstore:  %s\n\n", a.query, a.k, a.load_path);
        printf("Top %d matches:\n", hit_count);
        for (int i = 0; i < hit_count; i++) {
            printf("%d) score=%.4f chunk=%d\n%s\n\n",
                   i + 1, hits[i].score, hits[i].index, hits[i].text);
        }

        /* Build context string from hits (two-pass: measure then fill). */
        size_t total = 1;
        for (int i = 0; i < hit_count; i++) {
            total += strlen(hits[i].text);
            if (i < hit_count - 1) total += 3; /* "\n\n\n" */
        }
        char *context = malloc(total);
        if (context) {
            context[0] = '\0';
            for (int i = 0; i < hit_count; i++) {
                strcat(context, hits[i].text);
                if (i < hit_count - 1) strcat(context, "\n\n\n");
            }
        }

        generate_and_print(a.server_url, a.query, context);

        free(context);
        store_v1_free_hits(hits, hit_count);
        embed_free(ectx);
        return 0;
    }

    /* --------------------------------------------------------------------
     * Modes 1 and 3 both need the document chunked and embedded.
     * -------------------------------------------------------------------- */
    long file_len = 0;
    char *text = read_entire_file(a.file_path, &file_len);
    if (!text) {
        fprintf(stderr, "error: failed to read file: %s\n", a.file_path);
        embed_free(ectx);
        return 1;
    }

    int num_chunks = 0;
    char **chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP, &num_chunks);
    if (!chunks) {
        fprintf(stderr, "error: chunk_text failed\n");
        free(text);
        embed_free(ectx);
        return 1;
    }

    float *emb = embed_batch(ectx, (const char **)chunks, num_chunks);
    if (!emb) {
        fprintf(stderr, "error: embed_batch failed\n");
        chunk_free(chunks, num_chunks);
        free(text);
        embed_free(ectx);
        return 1;
    }

    printf("model:  %s\nfile:   %s\nbytes:  %ld\n", a.model_path, a.file_path, file_len);
    printf("chunks: %d\ndim:    %d\n", num_chunks, dim);

    /* ====================================================================
     * MODE 1: --file + --save
     * Write binary store to disk, then exit.
     * ==================================================================== */
    if (a.save_path) {
        if (store_v1_write(a.save_path, chunks, num_chunks, emb, dim) != 0) {
            fprintf(stderr, "error: store_v1_write failed\n");
            free(emb);
            chunk_free(chunks, num_chunks);
            free(text);
            embed_free(ectx);
            return 1;
        }
        printf("store:  %s (written)\n", a.save_path);
        free(emb);
        chunk_free(chunks, num_chunks);
        free(text);
        embed_free(ectx);
        return 0;
    }

    /* ====================================================================
     * MODE 3: --file + --query (in-memory, no disk store)
     * ==================================================================== */
    if (a.query) {
        printf("query:  %s\nk:      %d\n", a.query, a.k);

        int nprint = dim < 5 ? dim : 5;
        printf("The Embeddings as follows\n");
        printf("--------------------------\n");
        if (num_chunks > 0) print_embedding_preview(emb, 0, dim, nprint);
        if (num_chunks > 1) print_embedding_preview(emb, 1, dim, nprint);

        char *context = NULL;
        if (retrieve_topk(ectx, a.query, a.k, emb, num_chunks, dim, chunks, &context) != 0) {
            fprintf(stderr, "error: retrieval failed\n");
            free(emb);
            embed_free(ectx);
            chunk_free(chunks, num_chunks);
            free(text);
            return 1;
        }

        generate_and_print(a.server_url, a.query, context);
        free(context);
    }

    /* --------------------------------------------------------------------
     * Cleanup.
     * -------------------------------------------------------------------- */
    free(emb);
    embed_free(ectx);
    chunk_free(chunks, num_chunks);
    free(text);
    return 0;
}


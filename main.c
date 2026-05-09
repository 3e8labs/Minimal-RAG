#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chunk.h"
#include "embed.h"

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s --model <path.gtemodel> --file <doc.txt> [--query <text>] [--k <int>]\n", prog);
}

static char *read_entire_file(const char *path, long *len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    long n = ftell(fp);
    if (n < 0) { fclose(fp); return NULL; }
    if (fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return NULL; }

    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(fp); return NULL; }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); free(buf); return NULL; }
    fclose(fp);
    buf[n] = '\0';
    if (len) *len = n;
    return buf;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *file_path = NULL;

    const char *query = NULL;
    int k = 3;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--file") && i + 1 < argc) file_path = argv[++i];
        else if (!strcmp(argv[i], "--query") && i + 1 < argc) query = argv[++i];
        else if (!strcmp(argv[i], "--k") && i + 1 < argc) k = atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }

    if (!model_path || !file_path) { usage(argv[0]); return 1; }
    if (k <= 0) { fprintf(stderr, "error: --k must be > 0\n"); return 1; }

    long file_len = 0;
    char *text = read_entire_file(file_path, &file_len);
    if (!text) {
        fprintf(stderr, "error: failed to read file: %s\n", file_path);
        return 1;
    }

    int num_chunks = 0;
    char **chunks = chunk_text(text, 1000, 200, &num_chunks);
    if (!chunks) {
        fprintf(stderr, "error: chunk_text failed\n");
        free(text);
        return 1;
    }

    embed_ctx *ectx = embed_load(model_path);
    if (!ectx) {
        fprintf(stderr, "error: embed_load failed (model: %s)\n", model_path);
        chunk_free(chunks, num_chunks);
        free(text);
        return 1;
    }

    float *emb = embed_batch(ectx, (const char **)chunks, num_chunks);
    if (!emb) {
        fprintf(stderr, "error: embed_batch failed\n");
        embed_free(ectx);
        chunk_free(chunks, num_chunks);
        free(text);
        return 1;
    }

    int dim = embed_dim(ectx);
    printf("model:  %s\nfile:   %s\nbytes:  %ld\n", model_path, file_path, file_len);
    printf("chunks: %d\ndim:    %d\n", num_chunks, dim);
    if (query) {
        printf("query:  %s\nk:      %d\n", query, k);
        printf("note: query retrieval not implemented yet (next step)\n");
    }
    if (num_chunks > 0 && dim > 0) {
        int nprint = dim < 5 ? dim : 5;
        printf("emb[0][0..%d):", nprint);
        for (int i = 0; i < nprint; i++) {
          printf(" %.6f", emb[i]);
        }
        printf("\n");
    }

    if (num_chunks > 1 && dim > 0) {
        int nprint = dim < 5 ? dim : 5;
        printf("emb[1][0..%d):", nprint);
        for (int i = 0; i < nprint; i++) {
          printf(" %.6f", emb[dim + i]);
        }
        printf("\n");
    }

    free(emb);
    embed_free(ectx);
    chunk_free(chunks, num_chunks);
    free(text);
    return 0;
}


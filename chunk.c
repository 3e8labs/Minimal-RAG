#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chunk.h"

char **chunk_text(const char *text, int chunk_size, int overlap, int *num_chunks) {
    /* Validate inputs */
    if (!text) {
        fprintf(stderr, "chunk_text: text is NULL\n");
        return NULL;
    }
    if (chunk_size == 0) {
        fprintf(stderr, "chunk_text: chunk_size cannot be 0\n");
        return NULL;
    }
    if (overlap >= chunk_size) {
        fprintf(stderr, "chunk_text: overlap must be less than chunk_size\n");
        return NULL;
    }

    int text_len = strlen(text);
    int step = chunk_size - overlap;
    // step is just an internal metric used by memcpy, which basically represents the start of next chunk
    // It is a just a convenience variable.
    int count = (text_len - overlap) / step + 1;

    /* Allocate array of chunk pointers */
    char **chunks = malloc(count * sizeof(char *));
    if (!chunks) {
        fprintf(stderr, "chunk_text: out of memory\n");
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        int start = i * step; // Represents the start of the chunk
        int len = chunk_size;
        if (start + len > text_len) len = text_len - start;

        chunks[i] = malloc(len + 1);
        if (!chunks[i]) {
            fprintf(stderr, "chunk_text: out of memory\n");
            /* Free what we already allocated */
            for (int j = 0; j < i; j++) free(chunks[j]);
            free(chunks);
            return NULL;
        }

        memcpy(chunks[i], text + start, len);
        chunks[i][len] = '\0';
    }

    *num_chunks = count;
    return chunks;
}

void chunk_free(char **chunks, int num_chunks) {
    if (!chunks) return;
    for (int i = 0; i < num_chunks; i++) free(chunks[i]);
    free(chunks);
}

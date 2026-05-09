#ifndef CHUNK_H
#define CHUNK_H

char **chunk_text(const char *text, int chunk_size, int overlap, int *num_chunks);
void chunk_free(char **chunks, int num_chunks);

#endif

#ifndef CHUNK_H
#define CHUNK_H

char **chunk_text(const char *text, int chunk_size, int overlap, int *num_chunks);
// This chunk_text takes in a input query (user) as char *text.A
// Overlap is the chunk overlap between two chunks.
// const char * text = "abcdefgh";
// For example, the chunk_size = 4 , overlap =2
// Chunk -1 abcd
// Chunk -2 cdef
// Chunk -3 efgh
// Overlap is responsible for not losing information between two chunks.
// Number of chunks.
// Mallocs chunk array.
// Returns the array of chunks
void chunk_free(char **chunks, int num_chunks);
// Takes in array of chunks 
// Free. 
// And returns void

#endif

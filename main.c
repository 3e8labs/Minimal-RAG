#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s --model <path.gtemodel> --file <doc.txt>\n", prog);
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

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--file") && i + 1 < argc) file_path = argv[++i];
        else { usage(argv[0]); return 1; }
    }

    if (!model_path || !file_path) { usage(argv[0]); return 1; }

    long file_len = 0;
    char *text = read_entire_file(file_path, &file_len);
    if (!text) {
        fprintf(stderr, "error: failed to read file: %s\n", file_path);
        return 1;
    }

    printf("model: %s\nfile:  %s\nbytes: %ld\n", model_path, file_path, file_len);
    free(text);
    return 0;
}


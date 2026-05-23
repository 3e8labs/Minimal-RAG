#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>

#include "llm.h"

struct llm_ctx {
    char *server_url;
};

llm_ctx *llm_load(const char *server_url) {
    if (!server_url) {
        fprintf(stderr, "llm_load: server_url is NULL\n");
        return NULL;
    }

    llm_ctx *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "llm_load: out of memory\n");
        return NULL;
    }

    ctx->server_url = strdup(server_url);
    if (!ctx->server_url) {
        fprintf(stderr, "llm_load: out of memory\n");
        free(ctx);
        return NULL;
    }

    return ctx;
}

void llm_free(llm_ctx *ctx) {
    if (!ctx) return;
    free(ctx->server_url);
    free(ctx);
}

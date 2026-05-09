#include "embed.h"

#include <stdlib.h>

#include "third_party/gte/gte.h"

struct embed_ctx {
    gte_ctx *gte;
};

embed_ctx *embed_load(const char *model_path) {
    if (!model_path) return NULL;

    gte_ctx *gte = gte_load(model_path);
    if (!gte) return NULL;

    embed_ctx *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        gte_free(gte);
        return NULL;
    }

    ctx->gte = gte;
    return ctx;
}

void embed_free(embed_ctx *ctx) {
    if (!ctx) return;
    gte_free(ctx->gte);
    free(ctx);
}

float *embed_batch(embed_ctx *ctx, const char **texts, int count) {
    if (!ctx || !ctx->gte) return NULL;
    if (!texts || count <= 0) return NULL;

    return gte_embed_batch(ctx->gte, texts, count);
}

int embed_dim(embed_ctx *ctx) {
    if (!ctx || !ctx->gte) return 0;
    return gte_dim(ctx->gte);
}

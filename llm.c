#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>

#include "llm.h"

struct llm_ctx {
    char *server_url;
};

/* Replace newlines with spaces so the string is safe to embed in JSON. */
static char *flatten(const char *s) {
    int len = strlen(s);
    char *out = malloc(len + 1);
    if (!out) return NULL;
    for (int i = 0; i < len; i++)
        out[i] = s[i] == '\n' ? ' ' : s[i];
    out[len] = '\0';
    return out;
}

/* Growing buffer to accumulate libcurl response data. */
typedef struct {
    char  *data;  
    size_t len; 
    size_t cap; 
} strbuf;

/* libcurl calls this each time response bytes arrive.
 * We append to strbuf, doubling capacity when needed. */
static size_t on_data(void *ptr, size_t size, size_t nmemb, void *userdata) {
/* Input function set from the curl easy option CURLOPT_WRITEFUNCTION 
 * Outputs no.of bytes that are recieved. If nmemb != return value of callback func , signals libcurl that there is an error.
 * size is always 1.
 * Refer `https://curl.se/libcurl/c/CURLOPT_WRITEFUNCTION.html`
*/
  
    size_t bytes = size * nmemb; 
    strbuf *buf = (strbuf*)userdata; // First append the userdata to buf.

    /* Grow if needed */
    if (buf->len + bytes + 1 > buf->cap) {
        size_t new_cap = buf->cap == 0 ? 1024 : buf->cap * 2; // 1024 is the default capacity, otherwise double it.
        while (new_cap < buf->len + bytes + 1) new_cap *= 2;
        char *new_data = realloc(buf->data, new_cap);
        if (!new_data) return 0; /* returning 0 signals error to libcurl */
        buf->data = new_data;
        buf->cap = new_cap;
    }

    memcpy(buf->data + buf->len, ptr, bytes);
    buf->len += bytes;
    buf->data[buf->len] = '\0';
    return bytes;
}

llm_ctx *llm_load(const char *server_url) {
  // Input : server_url of the llama.cpp server
  // Returns : llm_ctx
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
    // Free-ing the llm context
    if (!ctx) return;
    free(ctx->server_url);
    free(ctx);
}

char *llm_generate(llm_ctx *ctx, const char *query, const char *context) {
    if (!ctx || !query || !context) {
        fprintf(stderr, "llm_generate: invalid arguments\n");
        return NULL;
    }

    /* ----------------------------------------------------------------
     * Build the URL.
     * Print to `url` replacing the string ctx->server_url and appending the API endpoint.
     * ---------------------------------------------------------------- */
    char url[256];
    snprintf(url, sizeof(url), "%s/v1/chat/completions", ctx->server_url);

    /* ----------------------------------------------------------------
     * Build the JSON request body.
     * Template accounts for all fixed JSON scaffolding (~512 bytes).
     * ---------------------------------------------------------------- */
    static const char *tmpl =
        "{"
        "\"model\":\"tinyllama\","
        "\"messages\":["
        "{\"role\":\"system\",\"content\":\"You are a helpful assistant. Use the following context to answer the question.\"},"
        "{\"role\":\"user\",\"content\":\"Context:\\n%s\\n\\nQuestion: %s\\nAnswer:\"}"
        "]}";

    char *flat_context = flatten(context);
    char *flat_query   = flatten(query);
    if (!flat_context || !flat_query) {
        fprintf(stderr, "llm_generate: out of memory\n");
        free(flat_context);
        free(flat_query);
        return NULL;
    }

    int body_len = strlen(flat_context) + strlen(flat_query) + 512;
    char *body = malloc(body_len + 1);
    if (!body) {
        fprintf(stderr, "llm_generate: out of memory\n");
        free(flat_context);
        free(flat_query);
        return NULL;
    }
    snprintf(body, body_len + 1, tmpl, flat_context, flat_query);
    free(flat_context);
    free(flat_query);

    /* ----------------------------------------------------------------
     * Set up libcurl and perform the HTTP POST.
     * ---------------------------------------------------------------- */
    strbuf response = {0};

    CURL *curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "llm_generate: curl_easy_init failed\n");
        free(body);
        return NULL;
    }

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, on_data); 
    // Callback function set through setopt CURLOPT_WRITEFUNCTION
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response); 

    CURLcode res = curl_easy_perform(curl); // Actual Network response

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    free(body);

    if (res != CURLE_OK) {
        fprintf(stderr, "llm_generate: curl error: %s\n", curl_easy_strerror(res));
        free(response.data);
        return NULL;
    }

    /* ----------------------------------------------------------------
     * Parse the JSON response — extract "content" value manually.
     * ---------------------------------------------------------------- */
    if (!response.data) {
        fprintf(stderr, "llm_generate: empty response\n");
        return NULL;
    }

    char *p = strstr(response.data, "\"content\":"); // Finding the content 
    if (!p) {
        fprintf(stderr, "llm_generate: no content field in response\n");
        free(response.data);
        return NULL;
    }

    /* Skip past "content":" (10 chars for key, then the opening quote) */
    p += 10;
    while (*p == ' ' || *p == ':' || *p == '"') p++;

    /* Walk to end of content value, handling escaped quotes */
    char *end = p;
    while (*end) {
        if (*end == '\\' && *(end + 1) == '"') end += 2;
        else if (*end == '"') break;
        else end++;
    }

    /* Copy and unescape \" → " */
    int answer_len = end - p;
    char *answer = malloc(answer_len + 1);
    if (!answer) {
        free(response.data);
        return NULL;
    }

    char *src = p;
    char *dst = answer;
    while (src < p + answer_len) {
        if (*src == '\\' && *(src + 1) == '"') { *dst++ = '"'; src += 2; }
        else *dst++ = *src++;
    }
    *dst = '\0';

    free(response.data);
    return answer;
}

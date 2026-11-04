/*
 * esp32-llm - llama2.c v2 int8 inference, portable C for ESP32.
 *
 * Original author: Andrej Karpathy
 *   llama2.c / runq.c - https://github.com/karpathy/llama2.c
 *   Copyright (c) 2023 Andrej Karpathy, MIT License.
 *
 * Modified for ESP32: no file I/O, no Arduino or ESP-IDF dependencies, a
 * caller-provided allocator, a streaming token API, and all mutable state in
 * ellm_model instead of globals.
 */

#include "esp32-llm.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ELLM_MAGIC 0x616B3432u
#define ELLM_HEADER_SIZE 256
#define ELLM_NORM_EPS 1e-5f
#define ELLM_ROPE_THETA 10000.0f
#define ELLM_BOS 1
#define ELLM_EOS 2
#define ELLM_MAX_TOKEN_LEN 4096

typedef struct {
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t vocab_size;
    int32_t seq_len;
} ellm_config;

typedef struct {
    int8_t* q;
    float* s;
} ellm_qmat;

typedef struct {
    ellm_qmat* q_tokens;
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;
    ellm_qmat* wq;
    ellm_qmat* wk;
    ellm_qmat* wv;
    ellm_qmat* wo;
    ellm_qmat* w1;
    ellm_qmat* w2;
    ellm_qmat* w3;
    float* rms_final_weight;
    ellm_qmat* wcls;
} ellm_weights;

typedef struct {
    float* x;
    float* xb;
    float* xb2;
    float* hb;
    float* hb2;
    ellm_qmat xq;
    ellm_qmat hq;
    float* q;
    float* k;
    float* v;
    float* att;
    float* logits;
    float* key_cache;
    float* value_cache;
} ellm_run_state;

typedef struct {
    char* str;
    int32_t id;
} ellm_token_index;

typedef struct {
    char** vocab;
    float* vocab_scores;
    char* vocab_blob;
    ellm_token_index* sorted_vocab;
    int32_t vocab_size;
    uint32_t max_token_length;
    unsigned char byte_pieces[512];
} ellm_tokenizer;

typedef struct {
    float prob;
    int32_t index;
} ellm_probidx;

typedef struct {
    int32_t vocab_size;
    ellm_probidx* probindex;
    float temperature;
    float topp;
    uint64_t rng_state;
} ellm_sampler;

struct ellm_model {
    ellm_config c;
    ellm_weights w;
    ellm_run_state s;
    ellm_tokenizer tok;
    ellm_sampler sampler;

    const uint8_t* model_data;
    size_t model_size;
    int32_t gs;

    float* rope_cos;
    float* rope_sin;
    int32_t rope_half;

    int32_t* prompt_tokens;
    int32_t n_prompt;
    int32_t max_new;
    int32_t pos;
    int32_t token;
    int32_t generated;
    int running;

    char* str_buffer;
    int dequantize_embeddings;

    void* (*alloc_fn)(size_t, void*);
    void (*free_fn)(void*, void*);
    void* alloc_ctx;
};

static void* ellm_default_alloc(size_t n, void* ctx) {
    (void)ctx;
    return malloc(n);
}

static void ellm_default_free(void* p, void* ctx) {
    (void)ctx;
    free(p);
}

static void* ellm_malloc(ellm_model* m, size_t n) {
    if (n == 0) {
        n = 1;
    }
    return m->alloc_fn(n, m->alloc_ctx);
}

static void* ellm_calloc(ellm_model* m, size_t n) {
    void* p = ellm_malloc(m, n);
    if (p) {
        memset(p, 0, n);
    }
    return p;
}

static void ellm_mfree(ellm_model* m, void* p) {
    if (p) {
        m->free_fn(p, m->alloc_ctx);
    }
}

const char* ellm_status_str(ellm_status status) {
    switch (status) {
        case ELLM_OK: return "ok";
        case ELLM_ERR_ARG: return "invalid argument";
        case ELLM_ERR_MEM: return "out of memory";
        case ELLM_ERR_FORMAT: return "invalid model or tokenizer format";
        case ELLM_ERR_STATE: return "invalid state";
        case ELLM_ERR_PROMPT: return "prompt too long or empty";
        default: return "unknown error";
    }
}

static int ellm_add_matrix(uint64_t* total, uint64_t n, int32_t gs) {
    if (gs <= 0 || n == 0) {
        return -1;
    }
    if (n % (uint64_t)gs != 0 || n % 4u != 0) {
        return -1;
    }
    *total += n + 4u * (n / (uint64_t)gs);
    return *total > (uint64_t)SIZE_MAX ? -1 : 0;
}

static int ellm_expected_size(const ellm_config* c, int32_t gs, int shared, size_t* out) {
    uint64_t total = ELLM_HEADER_SIZE;
    uint64_t n;
    int32_t head_dim = c->dim / c->n_heads;

    total += (uint64_t)(2 * c->n_layers + 1) * (uint64_t)c->dim * 4u;

    n = (uint64_t)c->vocab_size * (uint64_t)c->dim;
    if (ellm_add_matrix(&total, n, gs) != 0) {
        return -1;
    }

    n = (uint64_t)c->dim * (uint64_t)c->n_heads * (uint64_t)head_dim;
    for (int32_t i = 0; i < c->n_layers; i++) {
        if (ellm_add_matrix(&total, n, gs) != 0) {
            return -1;
        }
    }

    n = (uint64_t)c->dim * (uint64_t)c->n_kv_heads * (uint64_t)head_dim;
    for (int32_t i = 0; i < 2 * c->n_layers; i++) {
        if (ellm_add_matrix(&total, n, gs) != 0) {
            return -1;
        }
    }

    n = (uint64_t)c->n_heads * (uint64_t)head_dim * (uint64_t)c->dim;
    for (int32_t i = 0; i < c->n_layers; i++) {
        if (ellm_add_matrix(&total, n, gs) != 0) {
            return -1;
        }
    }

    n = (uint64_t)c->dim * (uint64_t)c->hidden_dim;
    for (int32_t i = 0; i < 3 * c->n_layers; i++) {
        if (ellm_add_matrix(&total, n, gs) != 0) {
            return -1;
        }
    }

    if (!shared) {
        n = (uint64_t)c->dim * (uint64_t)c->vocab_size;
        if (ellm_add_matrix(&total, n, gs) != 0) {
            return -1;
        }
    }

    if (total > (uint64_t)SIZE_MAX) {
        return -1;
    }
    *out = (size_t)total;
    return 0;
}

static int ellm_config_valid(const ellm_config* c) {
    if (c->dim <= 0 || c->hidden_dim <= 0 || c->n_layers <= 0 ||
        c->n_heads <= 0 || c->n_kv_heads <= 0 || c->vocab_size < 2 || c->seq_len <= 0) {
        return 0;
    }
    if (c->dim % c->n_heads != 0 || c->n_heads % c->n_kv_heads != 0) {
        return 0;
    }
    if ((c->dim / c->n_heads) % 2 != 0) {
        return 0;
    }
    if (c->n_layers > 100000 || c->vocab_size > 1000000 || c->seq_len > 1000000) {
        return 0;
    }
    return 1;
}

static void ellm_dequantize(const ellm_qmat* qx, float* x, int32_t n, int32_t gs) {
    for (int32_t i = 0; i < n; i++) {
        x[i] = (float)qx->q[i] * qx->s[i / gs];
    }
}

static void ellm_quantize(ellm_qmat* qx, const float* x, int32_t n, int32_t gs) {
    int32_t groups = n / gs;
    for (int32_t g = 0; g < groups; g++) {
        float wmax = 0.0f;
        for (int32_t i = 0; i < gs; i++) {
            float v = fabsf(x[g * gs + i]);
            if (v > wmax) {
                wmax = v;
            }
        }
        float scale = wmax / 127.0f;
        if (scale == 0.0f) {
            scale = 1.0f;
        }
        qx->s[g] = scale;
        for (int32_t i = 0; i < gs; i++) {
            qx->q[g * gs + i] = (int8_t)roundf(x[g * gs + i] / scale);
        }
    }
}

static ellm_qmat* ellm_init_qmats(ellm_model* m, const uint8_t** ptr, int32_t count, size_t size_each) {
    ellm_qmat* res = (ellm_qmat*)ellm_malloc(m, (size_t)count * sizeof(ellm_qmat));
    if (!res) {
        return NULL;
    }
    const uint8_t* p = *ptr;
    for (int32_t i = 0; i < count; i++) {
        res[i].q = (int8_t*)p;
        p += size_each;
        res[i].s = (float*)p;
        p += (size_each / (size_t)m->gs) * sizeof(float);
    }
    *ptr = p;
    return res;
}

static ellm_status ellm_map_weights(ellm_model* m, const uint8_t* ptr, int shared) {
    ellm_weights* w = &m->w;
    const ellm_config* c = &m->c;
    int32_t head_dim = c->dim / c->n_heads;

    float* fptr = (float*)ptr;
    w->rms_att_weight = fptr;
    fptr += c->n_layers * c->dim;
    w->rms_ffn_weight = fptr;
    fptr += c->n_layers * c->dim;
    w->rms_final_weight = fptr;
    fptr += c->dim;

    const uint8_t* p = (const uint8_t*)fptr;
    w->q_tokens = ellm_init_qmats(m, &p, 1, (size_t)c->vocab_size * (size_t)c->dim);
    if (!w->q_tokens) {
        return ELLM_ERR_MEM;
    }

    if (m->dequantize_embeddings) {
        w->token_embedding_table = (float*)ellm_malloc(m, (size_t)c->vocab_size * (size_t)c->dim * sizeof(float));
        if (!w->token_embedding_table) {
            return ELLM_ERR_MEM;
        }
        ellm_dequantize(w->q_tokens, w->token_embedding_table, c->vocab_size * c->dim, m->gs);
    }

    w->wq = ellm_init_qmats(m, &p, c->n_layers, (size_t)c->dim * (size_t)c->n_heads * (size_t)head_dim);
    w->wk = ellm_init_qmats(m, &p, c->n_layers, (size_t)c->dim * (size_t)c->n_kv_heads * (size_t)head_dim);
    w->wv = ellm_init_qmats(m, &p, c->n_layers, (size_t)c->dim * (size_t)c->n_kv_heads * (size_t)head_dim);
    w->wo = ellm_init_qmats(m, &p, c->n_layers, (size_t)c->n_heads * (size_t)head_dim * (size_t)c->dim);
    w->w1 = ellm_init_qmats(m, &p, c->n_layers, (size_t)c->dim * (size_t)c->hidden_dim);
    w->w2 = ellm_init_qmats(m, &p, c->n_layers, (size_t)c->hidden_dim * (size_t)c->dim);
    w->w3 = ellm_init_qmats(m, &p, c->n_layers, (size_t)c->dim * (size_t)c->hidden_dim);

    if (!w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3) {
        return ELLM_ERR_MEM;
    }

    if (shared) {
        w->wcls = w->q_tokens;
    } else {
        w->wcls = ellm_init_qmats(m, &p, 1, (size_t)c->dim * (size_t)c->vocab_size);
        if (!w->wcls) {
            return ELLM_ERR_MEM;
        }
    }
    return ELLM_OK;
}

static ellm_status ellm_alloc_run_state(ellm_model* m) {
    ellm_run_state* s = &m->s;
    const ellm_config* c = &m->c;
    int32_t kv_dim = (c->dim * c->n_kv_heads) / c->n_heads;

    s->x = (float*)ellm_calloc(m, (size_t)c->dim * sizeof(float));
    s->xb = (float*)ellm_calloc(m, (size_t)c->dim * sizeof(float));
    s->xb2 = (float*)ellm_calloc(m, (size_t)c->dim * sizeof(float));
    s->hb = (float*)ellm_calloc(m, (size_t)c->hidden_dim * sizeof(float));
    s->hb2 = (float*)ellm_calloc(m, (size_t)c->hidden_dim * sizeof(float));
    s->xq.q = (int8_t*)ellm_calloc(m, (size_t)c->dim);
    s->xq.s = (float*)ellm_calloc(m, (size_t)c->dim * sizeof(float));
    s->hq.q = (int8_t*)ellm_calloc(m, (size_t)c->hidden_dim);
    s->hq.s = (float*)ellm_calloc(m, (size_t)c->hidden_dim * sizeof(float));
    s->q = (float*)ellm_calloc(m, (size_t)c->dim * sizeof(float));
    s->k = (float*)ellm_calloc(m, (size_t)kv_dim * sizeof(float));
    s->v = (float*)ellm_calloc(m, (size_t)kv_dim * sizeof(float));
    s->att = (float*)ellm_calloc(m, (size_t)c->n_heads * (size_t)c->seq_len * sizeof(float));
    s->logits = (float*)ellm_calloc(m, (size_t)c->vocab_size * sizeof(float));
    s->key_cache = (float*)ellm_calloc(m, (size_t)c->n_layers * (size_t)c->seq_len * (size_t)kv_dim * sizeof(float));
    s->value_cache = (float*)ellm_calloc(m, (size_t)c->n_layers * (size_t)c->seq_len * (size_t)kv_dim * sizeof(float));

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->xq.q || !s->xq.s ||
        !s->hq.q || !s->hq.s || !s->q || !s->k || !s->v || !s->att || !s->logits ||
        !s->key_cache || !s->value_cache) {
        return ELLM_ERR_MEM;
    }
    return ELLM_OK;
}

static ellm_status ellm_build_rope(ellm_model* m) {
    int32_t head_dim = m->c.dim / m->c.n_heads;
    m->rope_half = head_dim / 2;
    size_t count = (size_t)m->c.seq_len * (size_t)m->rope_half;
    m->rope_cos = (float*)ellm_malloc(m, count * sizeof(float));
    m->rope_sin = (float*)ellm_malloc(m, count * sizeof(float));
    if (!m->rope_cos || !m->rope_sin) {
        return ELLM_ERR_MEM;
    }
    for (int32_t pos = 0; pos < m->c.seq_len; pos++) {
        for (int32_t j = 0; j < m->rope_half; j++) {
            float freq = 1.0f / powf(ELLM_ROPE_THETA, (2.0f * (float)j) / (float)head_dim);
            float val = (float)pos * freq;
            m->rope_cos[pos * m->rope_half + j] = cosf(val);
            m->rope_sin[pos * m->rope_half + j] = sinf(val);
        }
    }
    return ELLM_OK;
}

static int ellm_compare_tokens(const void* a, const void* b) {
    const ellm_token_index* ta = (const ellm_token_index*)a;
    const ellm_token_index* tb = (const ellm_token_index*)b;
    return strcmp(ta->str, tb->str);
}

static ellm_status ellm_load_tokenizer(ellm_model* m, const uint8_t* data, size_t size) {
    ellm_tokenizer* t = &m->tok;
    int32_t V = m->c.vocab_size;
    uint32_t maxlen;
    size_t off = 4;
    size_t blob = 0;

    if (size < 4) {
        return ELLM_ERR_FORMAT;
    }
    memcpy(&maxlen, data, 4);
    if (maxlen == 0 || maxlen > ELLM_MAX_TOKEN_LEN) {
        return ELLM_ERR_FORMAT;
    }
    t->max_token_length = maxlen;
    t->vocab_size = V;
    for (int32_t i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    for (int32_t i = 0; i < V; i++) {
        int32_t len;
        if (off + 8 > size) {
            return ELLM_ERR_FORMAT;
        }
        off += 4;
        memcpy(&len, data + off, 4);
        off += 4;
        if (len < 0 || (uint32_t)len > ELLM_MAX_TOKEN_LEN) {
            return ELLM_ERR_FORMAT;
        }
        if (off + (size_t)len > size) {
            return ELLM_ERR_FORMAT;
        }
        off += (size_t)len;
        blob += (size_t)len + 1;
    }
    if (off != size) {
        return ELLM_ERR_FORMAT;
    }

    t->vocab = (char**)ellm_malloc(m, (size_t)V * sizeof(char*));
    t->vocab_scores = (float*)ellm_malloc(m, (size_t)V * sizeof(float));
    t->vocab_blob = (char*)ellm_malloc(m, blob);
    if (!t->vocab || !t->vocab_scores || !t->vocab_blob) {
        return ELLM_ERR_MEM;
    }

    off = 4;
    char* cursor = t->vocab_blob;
    for (int32_t i = 0; i < V; i++) {
        int32_t len;
        memcpy(&t->vocab_scores[i], data + off, 4);
        off += 4;
        memcpy(&len, data + off, 4);
        off += 4;
        t->vocab[i] = cursor;
        memcpy(cursor, data + off, (size_t)len);
        cursor[len] = '\0';
        cursor += (size_t)len + 1;
        off += (size_t)len;
    }

    t->sorted_vocab = (ellm_token_index*)ellm_malloc(m, (size_t)V * sizeof(ellm_token_index));
    if (!t->sorted_vocab) {
        return ELLM_ERR_MEM;
    }
    for (int32_t i = 0; i < V; i++) {
        t->sorted_vocab[i].str = t->vocab[i];
        t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, (size_t)V, sizeof(ellm_token_index), ellm_compare_tokens);

    m->str_buffer = (char*)ellm_malloc(m, (size_t)maxlen * 2 + 3);
    if (!m->str_buffer) {
        return ELLM_ERR_MEM;
    }
    return ELLM_OK;
}

static int32_t ellm_str_lookup(const char* str, const ellm_token_index* sorted, int32_t n) {
    ellm_token_index key;
    key.str = (char*)str;
    key.id = 0;
    const ellm_token_index* res = (const ellm_token_index*)bsearch(
        &key, sorted, (size_t)n, sizeof(ellm_token_index), ellm_compare_tokens);
    return res ? res->id : -1;
}

static int ellm_push_token(int32_t* tokens, int32_t* n, int32_t cap, int32_t id) {
    if (*n >= cap) {
        return -1;
    }
    tokens[(*n)++] = id;
    return 0;
}

static int ellm_encode(ellm_model* m, const char* text, int bos, int eos,
                       int32_t* tokens, int32_t cap, int32_t* n_tokens) {
    ellm_tokenizer* t = &m->tok;
    char* str_buffer = m->str_buffer;
    int32_t n = 0;
    size_t str_len = 0;

    if (bos && ellm_push_token(tokens, &n, cap, ELLM_BOS) != 0) {
        return -1;
    }

    if (text[0] != '\0') {
        int32_t dummy = ellm_str_lookup(" ", t->sorted_vocab, t->vocab_size);
        if (dummy < 0 || ellm_push_token(tokens, &n, cap, dummy) != 0) {
            return -1;
        }
    }

    for (const char* c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }
        int32_t id = ellm_str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            if (ellm_push_token(tokens, &n, cap, id) != 0) {
                return -1;
            }
        } else {
            for (size_t i = 0; i < str_len; i++) {
                int32_t bid = (unsigned char)str_buffer[i] + 3;
                if (bid >= t->vocab_size || ellm_push_token(tokens, &n, cap, bid) != 0) {
                    return -1;
                }
            }
        }
        str_len = 0;
    }

    for (;;) {
        float best_score = -1e10f;
        int32_t best_id = -1;
        int32_t best_idx = -1;
        for (int32_t i = 0; i < n - 1; i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int32_t id = ellm_str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) {
            break;
        }
        tokens[best_idx] = best_id;
        for (int32_t i = best_idx + 1; i < n - 1; i++) {
            tokens[i] = tokens[i + 1];
        }
        n--;
    }

    if (eos && ellm_push_token(tokens, &n, cap, ELLM_EOS) != 0) {
        return -1;
    }
    *n_tokens = n;
    return 0;
}

static char* ellm_decode(ellm_model* m, int32_t prev_token, int32_t token) {
    ellm_tokenizer* t = &m->tok;
    char* piece = t->vocab[token];
    unsigned char byte_val;

    if (prev_token == ELLM_BOS && piece[0] == ' ') {
        piece++;
    }
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

static void ellm_rmsnorm(float* o, const float* x, const float* weight, int32_t size) {
    float ss = 0.0f;
    for (int32_t j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= (float)size;
    ss += ELLM_NORM_EPS;
    ss = 1.0f / sqrtf(ss);
    for (int32_t j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

static void ellm_softmax(float* x, int32_t size) {
    float max_val = x[0];
    for (int32_t i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int32_t i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int32_t i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

static void ellm_matmul(float* xout, const ellm_qmat* x, const ellm_qmat* w,
                        int32_t n, int32_t d, int32_t gs) {
    for (int32_t i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int32_t in = i * n;
        for (int32_t j = 0; j <= n - gs; j += gs) {
            for (int32_t k = 0; k < gs; k++) {
                ival += ((int32_t)x->q[j + k]) * ((int32_t)w->q[in + j + k]);
            }
            val += ((float)ival) * w->s[(in + j) / gs] * x->s[j / gs];
            ival = 0;
        }
        xout[i] = val;
    }
}

static float* ellm_forward(ellm_model* m, int32_t token, int32_t pos) {
    const ellm_config* p = &m->c;
    const ellm_weights* w = &m->w;
    ellm_run_state* s = &m->s;
    float* x = s->x;
    int32_t dim = p->dim;
    int32_t kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int32_t kv_mul = p->n_heads / p->n_kv_heads;
    int32_t hidden_dim = p->hidden_dim;
    int32_t head_dim = dim / p->n_heads;

    if (w->token_embedding_table) {
        memcpy(x, w->token_embedding_table + (size_t)token * (size_t)dim,
               (size_t)dim * sizeof(float));
    } else {
        const int8_t* q = w->q_tokens->q + (size_t)token * (size_t)dim;
        const float* sc = w->q_tokens->s + ((size_t)token * (size_t)dim) / (size_t)m->gs;
        for (int32_t i = 0; i < dim; i++) {
            x[i] = (float)q[i] * sc[i / m->gs];
        }
    }

    for (int32_t l = 0; l < p->n_layers; l++) {
        ellm_rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        ellm_quantize(&s->xq, s->xb, dim, m->gs);
        ellm_matmul(s->q, &s->xq, w->wq + l, dim, dim, m->gs);
        ellm_matmul(s->k, &s->xq, w->wk + l, dim, kv_dim, m->gs);
        ellm_matmul(s->v, &s->xq, w->wv + l, dim, kv_dim, m->gs);

        for (int32_t i = 0; i < dim; i += 2) {
            int32_t j = (i % head_dim) >> 1;
            float fcr = m->rope_cos[pos * m->rope_half + j];
            float fci = m->rope_sin[pos * m->rope_half + j];
            int32_t rotn = i < kv_dim ? 2 : 1;
            for (int32_t v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        int32_t loff = l * p->seq_len * kv_dim;
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, (size_t)kv_dim * sizeof(float));
        memcpy(value_cache_row, s->v, (size_t)kv_dim * sizeof(float));

        for (int32_t h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_dim;
            float* att = s->att + h * p->seq_len;
            for (int32_t t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                float score = 0.0f;
                for (int32_t i = 0; i < head_dim; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf((float)head_dim);
                att[t] = score;
            }

            ellm_softmax(att, pos + 1);

            float* xb = s->xb + h * head_dim;
            memset(xb, 0, (size_t)head_dim * sizeof(float));
            for (int32_t t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                float a = att[t];
                for (int32_t i = 0; i < head_dim; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        ellm_quantize(&s->xq, s->xb, dim, m->gs);
        ellm_matmul(s->xb2, &s->xq, w->wo + l, dim, dim, m->gs);

        for (int32_t i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        ellm_rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        ellm_quantize(&s->xq, s->xb, dim, m->gs);
        ellm_matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim, m->gs);
        ellm_matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim, m->gs);

        for (int32_t i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        ellm_quantize(&s->hq, s->hb, hidden_dim, m->gs);
        ellm_matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim, m->gs);

        for (int32_t i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    ellm_rmsnorm(x, x, w->rms_final_weight, dim);
    ellm_quantize(&s->xq, x, dim, m->gs);
    ellm_matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size, m->gs);
    return s->logits;
}

static int32_t ellm_sample_argmax(const float* probabilities, int32_t n) {
    int32_t max_i = 0;
    float max_p = probabilities[0];
    for (int32_t i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

static int32_t ellm_sample_mult(const float* probabilities, int32_t n, float coin) {
    float cdf = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

static int ellm_compare_probidx(const void* a, const void* b) {
    const ellm_probidx* pa = (const ellm_probidx*)a;
    const ellm_probidx* pb = (const ellm_probidx*)b;
    if (pa->prob > pb->prob) {
        return -1;
    }
    if (pa->prob < pb->prob) {
        return 1;
    }
    return 0;
}

static int32_t ellm_sample_topp(const float* probabilities, int32_t n, float topp,
                                ellm_probidx* probindex, float coin) {
    int32_t n0 = 0;
    const float cutoff = (1.0f - topp) / (float)(n - 1);
    for (int32_t i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    if (n0 == 0) {
        return ellm_sample_argmax(probabilities, n);
    }
    qsort(probindex, (size_t)n0, sizeof(ellm_probidx), ellm_compare_probidx);

    float cumulative_prob = 0.0f;
    int32_t last_idx = n0 - 1;
    for (int32_t i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int32_t i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

static uint32_t ellm_random_u32(uint64_t* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (uint32_t)((*state * 0x2545F4914F6CDD1Dull) >> 32);
}

static float ellm_random_f32(uint64_t* state) {
    return (ellm_random_u32(state) >> 8) / 16777216.0f;
}

static int32_t ellm_sample(ellm_model* m, float* logits) {
    ellm_sampler* sampler = &m->sampler;
    int32_t next;
    if (sampler->temperature == 0.0f) {
        next = ellm_sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int32_t q = 0; q < sampler->vocab_size; q++) {
            logits[q] /= sampler->temperature;
        }
        ellm_softmax(logits, sampler->vocab_size);
        float coin = ellm_random_f32(&sampler->rng_state);
        if (sampler->topp <= 0.0f || sampler->topp >= 1.0f) {
            next = ellm_sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = ellm_sample_topp(logits, sampler->vocab_size, sampler->topp,
                                    sampler->probindex, coin);
        }
    }
    return next;
}

ellm_status ellm_load(const void* model_data, size_t model_size,
                      const void* tokenizer_data, size_t tokenizer_size,
                      const ellm_options* options, ellm_model** out_model) {
    const ellm_allocator* allocator;
    void* (*alloc_fn)(size_t, void*);
    void (*free_fn)(void*, void*);
    void* alloc_ctx;
    ellm_model* m;
    ellm_status st;
    size_t expected = 0;
    uint32_t magic;
    int32_t version;
    int32_t gs;
    uint8_t shared;

    if (!model_data || !tokenizer_data || !out_model) {
        return ELLM_ERR_ARG;
    }
    *out_model = NULL;
    if (model_size < ELLM_HEADER_SIZE) {
        return ELLM_ERR_FORMAT;
    }
    if (((uintptr_t)model_data & 3u) != 0) {
        return ELLM_ERR_ARG;
    }

    allocator = options ? options->allocator : NULL;
    if (allocator && (!allocator->alloc || !allocator->free)) {
        return ELLM_ERR_ARG;
    }
    alloc_fn = allocator ? allocator->alloc : ellm_default_alloc;
    free_fn = allocator ? allocator->free : ellm_default_free;
    alloc_ctx = allocator ? allocator->ctx : NULL;

    m = (ellm_model*)alloc_fn(sizeof(ellm_model), alloc_ctx);
    if (!m) {
        return ELLM_ERR_MEM;
    }
    memset(m, 0, sizeof(ellm_model));
    m->alloc_fn = alloc_fn;
    m->free_fn = free_fn;
    m->alloc_ctx = alloc_ctx;
    m->dequantize_embeddings = options ? (options->dequantize_embeddings ? 1 : 0) : 1;
    m->sampler.temperature = options ? options->temperature : 0.7f;
    m->sampler.topp = options ? options->top_p : 0.9f;
    m->sampler.rng_state = (options && options->seed) ? options->seed : 1;
    if (m->sampler.temperature < 0.0f) {
        m->sampler.temperature = 0.0f;
    }

    const uint8_t* data = (const uint8_t*)model_data;
    memcpy(&magic, data, 4);
    memcpy(&version, data + 4, 4);
    memcpy(&m->c, data + 8, sizeof(ellm_config));
    shared = data[36];
    memcpy(&gs, data + 37, 4);

    if (magic != ELLM_MAGIC || version != 2 || !ellm_config_valid(&m->c) || gs <= 0) {
        ellm_free(m);
        return ELLM_ERR_FORMAT;
    }
    m->gs = gs;
    if (m->c.dim % gs != 0 || m->c.hidden_dim % gs != 0) {
        ellm_free(m);
        return ELLM_ERR_FORMAT;
    }
    if (ellm_expected_size(&m->c, gs, shared ? 1 : 0, &expected) != 0 || expected != model_size) {
        ellm_free(m);
        return ELLM_ERR_FORMAT;
    }

    m->model_data = data;
    m->model_size = model_size;
    m->sampler.vocab_size = m->c.vocab_size;

    st = ellm_map_weights(m, data + ELLM_HEADER_SIZE, shared ? 1 : 0);
    if (st == ELLM_OK) {
        st = ellm_alloc_run_state(m);
    }
    if (st == ELLM_OK) {
        st = ellm_build_rope(m);
    }
    if (st == ELLM_OK) {
        st = ellm_load_tokenizer(m, (const uint8_t*)tokenizer_data, tokenizer_size);
    }
    if (st != ELLM_OK) {
        ellm_free(m);
        return st;
    }

    m->sampler.probindex = (ellm_probidx*)ellm_malloc(
        m, (size_t)m->c.vocab_size * sizeof(ellm_probidx));
    m->prompt_tokens = (int32_t*)ellm_malloc(
        m, (size_t)m->c.seq_len * sizeof(int32_t));
    if (!m->sampler.probindex || !m->prompt_tokens) {
        ellm_free(m);
        return ELLM_ERR_MEM;
    }

    *out_model = m;
    return ELLM_OK;
}

void ellm_free(ellm_model* m) {
    void (*free_fn)(void*, void*);
    void* alloc_ctx;

    if (!m) {
        return;
    }
    free_fn = m->free_fn;
    alloc_ctx = m->alloc_ctx;

    ellm_mfree(m, m->w.q_tokens);
    ellm_mfree(m, m->w.token_embedding_table);
    ellm_mfree(m, m->w.wq);
    ellm_mfree(m, m->w.wk);
    ellm_mfree(m, m->w.wv);
    ellm_mfree(m, m->w.wo);
    ellm_mfree(m, m->w.w1);
    ellm_mfree(m, m->w.w2);
    ellm_mfree(m, m->w.w3);
    if (m->w.wcls && m->w.wcls != m->w.q_tokens) {
        ellm_mfree(m, m->w.wcls);
    }

    ellm_mfree(m, m->s.x);
    ellm_mfree(m, m->s.xb);
    ellm_mfree(m, m->s.xb2);
    ellm_mfree(m, m->s.hb);
    ellm_mfree(m, m->s.hb2);
    ellm_mfree(m, m->s.xq.q);
    ellm_mfree(m, m->s.xq.s);
    ellm_mfree(m, m->s.hq.q);
    ellm_mfree(m, m->s.hq.s);
    ellm_mfree(m, m->s.q);
    ellm_mfree(m, m->s.k);
    ellm_mfree(m, m->s.v);
    ellm_mfree(m, m->s.att);
    ellm_mfree(m, m->s.logits);
    ellm_mfree(m, m->s.key_cache);
    ellm_mfree(m, m->s.value_cache);

    ellm_mfree(m, m->rope_cos);
    ellm_mfree(m, m->rope_sin);
    ellm_mfree(m, m->tok.vocab);
    ellm_mfree(m, m->tok.vocab_scores);
    ellm_mfree(m, m->tok.vocab_blob);
    ellm_mfree(m, m->tok.sorted_vocab);
    ellm_mfree(m, m->sampler.probindex);
    ellm_mfree(m, m->prompt_tokens);
    ellm_mfree(m, m->str_buffer);

    free_fn(m, alloc_ctx);
}

void ellm_set_sampling(ellm_model* m, float temperature, float top_p, uint64_t seed) {
    if (!m) {
        return;
    }
    m->sampler.temperature = temperature < 0.0f ? 0.0f : temperature;
    m->sampler.topp = top_p;
    if (seed) {
        m->sampler.rng_state = seed;
    }
}

ellm_status ellm_begin(ellm_model* m, const char* prompt, int max_new_tokens) {
    int32_t n = 0;
    if (!m || !prompt) {
        return ELLM_ERR_ARG;
    }
    if (m->running) {
        return ELLM_ERR_STATE;
    }
    if (ellm_encode(m, prompt, 1, 0, m->prompt_tokens, m->c.seq_len, &n) != 0) {
        return ELLM_ERR_PROMPT;
    }
    if (n < 1 || n >= m->c.seq_len) {
        return ELLM_ERR_PROMPT;
    }
    m->n_prompt = n;
    m->max_new = (max_new_tokens <= 0 || max_new_tokens > m->c.seq_len)
                     ? m->c.seq_len
                     : max_new_tokens;
    m->pos = 0;
    m->token = m->prompt_tokens[0];
    m->generated = 0;
    m->running = 1;
    return ELLM_OK;
}

ellm_step ellm_next(ellm_model* m, char* out, size_t out_cap, size_t* out_len) {
    int32_t next;
    if (!m || !out || !out_len) {
        return ELLM_STEP_ERROR;
    }
    *out_len = 0;
    if (!m->running) {
        return ELLM_STEP_DONE;
    }
    if (m->pos >= m->c.seq_len) {
        m->running = 0;
        return ELLM_STEP_DONE;
    }

    float* logits = ellm_forward(m, m->token, m->pos);
    if (m->pos < m->n_prompt - 1) {
        next = m->prompt_tokens[m->pos + 1];
    } else {
        next = ellm_sample(m, logits);
    }
    m->pos++;

    if (next == ELLM_BOS) {
        m->running = 0;
        return ELLM_STEP_DONE;
    }
    if (next == ELLM_EOS && m->pos >= m->n_prompt) {
        m->running = 0;
        return ELLM_STEP_DONE;
    }
    if (m->pos < m->n_prompt) {
        m->token = next;
        return ELLM_STEP_PROMPT;
    }

    char* piece = ellm_decode(m, m->token, next);
    size_t len = strlen(piece);
    if (len > out_cap) {
        len = out_cap;
    }
    memcpy(out, piece, len);
    *out_len = len;
    m->generated++;
    m->token = next;
    if (m->generated >= m->max_new) {
        m->running = 0;
    }
    return ELLM_STEP_TOKEN;
}

void ellm_abort(ellm_model* m) {
    if (m) {
        m->running = 0;
    }
}

ellm_status ellm_generate(ellm_model* m, const char* prompt, int max_new_tokens,
                          ellm_token_cb cb, void* ctx) {
    ellm_status result = ELLM_OK;
    size_t cap;
    char* buf;

    ellm_status st = ellm_begin(m, prompt, max_new_tokens);
    if (st != ELLM_OK) {
        return st;
    }
    cap = (size_t)m->tok.max_token_length + 1;
    buf = (char*)ellm_malloc(m, cap);
    if (!buf) {
        ellm_abort(m);
        return ELLM_ERR_MEM;
    }
    for (;;) {
        size_t len = 0;
        ellm_step step = ellm_next(m, buf, cap, &len);
        if (step == ELLM_STEP_TOKEN) {
            if (cb) {
                cb(buf, len, ctx);
            }
        } else if (step == ELLM_STEP_DONE) {
            break;
        } else if (step == ELLM_STEP_ERROR) {
            result = ELLM_ERR_STATE;
            break;
        }
    }
    ellm_mfree(m, buf);
    return result;
}

int ellm_vocab_size(const ellm_model* m) {
    return m ? (int)m->c.vocab_size : 0;
}

int ellm_context_size(const ellm_model* m) {
    return m ? (int)m->c.seq_len : 0;
}

size_t ellm_max_token_length(const ellm_model* m) {
    return m ? (size_t)m->tok.max_token_length : 0;
}

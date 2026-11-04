/*
 * esp32-llm - llama2.c v2 int8 inference, portable C for ESP32.
 *
 * Original author: Andrej Karpathy
 *   llama2.c / runq.c - https://github.com/karpathy/llama2.c
 *   Copyright (c) 2023 Andrej Karpathy, MIT License.
 *
 * Modified for ESP32: removed file I/O and Arduino dependencies, added a
 * caller-provided allocator and a streaming token API, and moved all mutable
 * state into ellm_model so these two files can be copied into any project.
 */
#ifndef ESP32_LLM_H
#define ESP32_LLM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ellm_model ellm_model;

typedef struct {
    void* (*alloc)(size_t bytes, void* ctx);
    void  (*free)(void* ptr, void* ctx);
    void* ctx;
} ellm_allocator;

typedef struct {
    const ellm_allocator* allocator;
    int dequantize_embeddings;
    float temperature;
    float top_p;
    uint64_t seed;
} ellm_options;

typedef enum {
    ELLM_OK = 0,
    ELLM_ERR_ARG,
    ELLM_ERR_MEM,
    ELLM_ERR_FORMAT,
    ELLM_ERR_STATE,
    ELLM_ERR_PROMPT
} ellm_status;

typedef enum {
    ELLM_STEP_PROMPT = 0,
    ELLM_STEP_TOKEN,
    ELLM_STEP_DONE,
    ELLM_STEP_ERROR
} ellm_step;

ellm_status ellm_load(const void* model_data, size_t model_size,
                      const void* tokenizer_data, size_t tokenizer_size,
                      const ellm_options* options, ellm_model** out_model);

void ellm_free(ellm_model* model);

void ellm_set_sampling(ellm_model* model, float temperature, float top_p, uint64_t seed);

ellm_status ellm_begin(ellm_model* model, const char* prompt, int max_new_tokens);
ellm_step ellm_next(ellm_model* model, char* out, size_t out_cap, size_t* out_len);
void ellm_abort(ellm_model* model);

typedef void (*ellm_token_cb)(const char* piece, size_t len, void* ctx);
ellm_status ellm_generate(ellm_model* model, const char* prompt, int max_new_tokens,
                          ellm_token_cb cb, void* ctx);

int ellm_vocab_size(const ellm_model* model);
int ellm_context_size(const ellm_model* model);
size_t ellm_max_token_length(const ellm_model* model);
const char* ellm_status_str(ellm_status status);

#ifdef __cplusplus
}
#endif

#endif

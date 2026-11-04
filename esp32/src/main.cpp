#include <Arduino.h>
#include <LittleFS.h>
#include <Wire.h>
#include <esp32-hal-psram.h>
#include <esp_random.h>
#include <esp_heap_caps.h>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "esp32-llm.h"

namespace {

constexpr int MAX_LINE = 300;

ellm_model* g_model = nullptr;
char* g_piece = nullptr;
size_t g_piece_cap = 0;
bool g_generating = false;
int g_generated = 0;
uint32_t g_start_ms = 0;
char g_prompt[MAX_LINE + 32];

constexpr uint8_t OLED_SDA = 8;
constexpr uint8_t OLED_SCL = 9;
constexpr uint8_t OLED_ADDR = 0x3C;
constexpr int OLED_W = 128;
constexpr int OLED_H = 64;
constexpr int OLED_COLS = OLED_W / 6;
constexpr int OLED_ROWS = OLED_H / 8;
constexpr int OLED_TEXT_ROWS = OLED_ROWS - 1;
constexpr int OLED_MAX_LINES = 64;
constexpr size_t OLED_TEXT_CAP = 1024;

static const uint8_t FONT5X7[96][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00},
    {0x00, 0x00, 0x5F, 0x00, 0x00},
    {0x00, 0x07, 0x00, 0x07, 0x00},
    {0x14, 0x7F, 0x14, 0x7F, 0x14},
    {0x24, 0x2A, 0x7F, 0x2A, 0x12},
    {0x23, 0x13, 0x08, 0x64, 0x62},
    {0x36, 0x49, 0x55, 0x22, 0x50},
    {0x00, 0x05, 0x03, 0x00, 0x00},
    {0x00, 0x1C, 0x22, 0x41, 0x00},
    {0x00, 0x41, 0x22, 0x1C, 0x00},
    {0x14, 0x08, 0x3E, 0x08, 0x14},
    {0x08, 0x08, 0x3E, 0x08, 0x08},
    {0x00, 0x50, 0x30, 0x00, 0x00},
    {0x08, 0x08, 0x08, 0x08, 0x08},
    {0x00, 0x60, 0x60, 0x00, 0x00},
    {0x20, 0x10, 0x08, 0x04, 0x02},
    {0x3E, 0x51, 0x49, 0x45, 0x3E},
    {0x00, 0x42, 0x7F, 0x40, 0x00},
    {0x42, 0x61, 0x51, 0x49, 0x46},
    {0x21, 0x41, 0x45, 0x4B, 0x31},
    {0x18, 0x14, 0x12, 0x7F, 0x10},
    {0x27, 0x45, 0x45, 0x45, 0x39},
    {0x3C, 0x4A, 0x49, 0x49, 0x30},
    {0x01, 0x71, 0x09, 0x05, 0x03},
    {0x36, 0x49, 0x49, 0x49, 0x36},
    {0x06, 0x49, 0x49, 0x29, 0x1E},
    {0x00, 0x36, 0x36, 0x00, 0x00},
    {0x00, 0x56, 0x36, 0x00, 0x00},
    {0x08, 0x14, 0x22, 0x41, 0x00},
    {0x14, 0x14, 0x14, 0x14, 0x14},
    {0x00, 0x41, 0x22, 0x14, 0x08},
    {0x02, 0x01, 0x51, 0x09, 0x06},
    {0x32, 0x49, 0x79, 0x41, 0x3E},
    {0x7E, 0x11, 0x11, 0x11, 0x7E},
    {0x7F, 0x49, 0x49, 0x49, 0x36},
    {0x3E, 0x41, 0x41, 0x41, 0x22},
    {0x7F, 0x41, 0x41, 0x22, 0x1C},
    {0x7F, 0x49, 0x49, 0x49, 0x41},
    {0x7F, 0x09, 0x09, 0x09, 0x01},
    {0x3E, 0x41, 0x49, 0x49, 0x7A},
    {0x7F, 0x08, 0x08, 0x08, 0x7F},
    {0x00, 0x41, 0x7F, 0x41, 0x00},
    {0x20, 0x40, 0x41, 0x3F, 0x01},
    {0x7F, 0x08, 0x14, 0x22, 0x41},
    {0x7F, 0x40, 0x40, 0x40, 0x40},
    {0x7F, 0x02, 0x0C, 0x02, 0x7F},
    {0x7F, 0x04, 0x08, 0x10, 0x7F},
    {0x3E, 0x41, 0x41, 0x41, 0x3E},
    {0x7F, 0x09, 0x09, 0x09, 0x06},
    {0x3E, 0x41, 0x51, 0x21, 0x5E},
    {0x7F, 0x09, 0x19, 0x29, 0x46},
    {0x46, 0x49, 0x49, 0x49, 0x31},
    {0x01, 0x01, 0x7F, 0x01, 0x01},
    {0x3F, 0x40, 0x40, 0x40, 0x3F},
    {0x1F, 0x20, 0x40, 0x20, 0x1F},
    {0x7F, 0x20, 0x18, 0x20, 0x7F},
    {0x63, 0x14, 0x08, 0x14, 0x63},
    {0x03, 0x04, 0x78, 0x04, 0x03},
    {0x61, 0x51, 0x49, 0x45, 0x43},
    {0x00, 0x7F, 0x41, 0x41, 0x00},
    {0x02, 0x04, 0x08, 0x10, 0x20},
    {0x00, 0x41, 0x41, 0x7F, 0x00},
    {0x04, 0x02, 0x01, 0x02, 0x04},
    {0x40, 0x40, 0x40, 0x40, 0x40},
    {0x00, 0x01, 0x02, 0x04, 0x00},
    {0x20, 0x54, 0x54, 0x54, 0x78},
    {0x7F, 0x48, 0x44, 0x44, 0x38},
    {0x38, 0x44, 0x44, 0x44, 0x20},
    {0x38, 0x44, 0x44, 0x48, 0x7F},
    {0x38, 0x54, 0x54, 0x54, 0x18},
    {0x08, 0x7E, 0x09, 0x01, 0x02},
    {0x0C, 0x52, 0x52, 0x52, 0x3E},
    {0x7F, 0x08, 0x04, 0x04, 0x78},
    {0x00, 0x44, 0x7D, 0x40, 0x00},
    {0x20, 0x40, 0x44, 0x3D, 0x00},
    {0x7F, 0x10, 0x28, 0x44, 0x00},
    {0x00, 0x41, 0x7F, 0x40, 0x00},
    {0x7C, 0x04, 0x18, 0x04, 0x78},
    {0x7C, 0x08, 0x04, 0x04, 0x78},
    {0x38, 0x44, 0x44, 0x44, 0x38},
    {0x7C, 0x14, 0x14, 0x14, 0x08},
    {0x08, 0x14, 0x14, 0x18, 0x7C},
    {0x7C, 0x08, 0x04, 0x04, 0x08},
    {0x48, 0x54, 0x54, 0x54, 0x20},
    {0x04, 0x3F, 0x44, 0x40, 0x20},
    {0x3C, 0x40, 0x40, 0x20, 0x7C},
    {0x1C, 0x20, 0x40, 0x20, 0x1C},
    {0x3C, 0x40, 0x30, 0x40, 0x3C},
    {0x44, 0x28, 0x10, 0x28, 0x44},
    {0x0C, 0x50, 0x50, 0x50, 0x3C},
    {0x44, 0x64, 0x54, 0x4C, 0x44},
    {0x00, 0x08, 0x36, 0x41, 0x00},
    {0x00, 0x00, 0x7F, 0x00, 0x00},
    {0x00, 0x41, 0x36, 0x08, 0x00},
    {0x08, 0x08, 0x2A, 0x1C, 0x08},
    {0x00, 0x00, 0x00, 0x00, 0x00}
};

bool g_oled_ok = false;
uint8_t g_oled_addr = OLED_ADDR;
uint8_t g_oled_buf[OLED_W * OLED_H / 8];
char g_oled_header[OLED_COLS + 1] = "esp32-llm";
char g_oled_text[OLED_TEXT_CAP];
size_t g_oled_text_len = 0;
char g_oled_lines[OLED_MAX_LINES][OLED_COLS + 1];
int g_oled_nlines = 0;

void oled_cmd(uint8_t c) {
    Wire.beginTransmission(g_oled_addr);
    Wire.write(0x00);
    Wire.write(c);
    Wire.endTransmission();
}

void oled_cmds(const uint8_t* cmds, size_t n) {
    Wire.beginTransmission(g_oled_addr);
    Wire.write(0x00);
    for (size_t i = 0; i < n; i++) {
        Wire.write(cmds[i]);
    }
    Wire.endTransmission();
}

bool oled_probe(uint8_t addr) {
    Wire.beginTransmission(addr);
    return Wire.endTransmission() == 0;
}

void oled_clear() {
    memset(g_oled_buf, 0, sizeof(g_oled_buf));
}

void oled_pixel(int x, int y) {
    if (x < 0 || x >= OLED_W || y < 0 || y >= OLED_H) {
        return;
    }
    g_oled_buf[x + (y / 8) * OLED_W] |= (uint8_t)(1u << (y & 7));
}

void oled_char(int col, int row, char ch) {
    if (ch < 32 || ch > 126) {
        ch = '?';
    }
    const uint8_t* glyph = FONT5X7[(int)ch - 32];
    int x0 = col * 6;
    int y0 = row * 8;
    for (int i = 0; i < 5; i++) {
        uint8_t bits = glyph[i];
        for (int j = 0; j < 7; j++) {
            if (bits & (1u << j)) {
                oled_pixel(x0 + i, y0 + j);
            }
        }
    }
}

void oled_line(int row, const char* text, int max_cols) {
    for (int col = 0; col < max_cols; col++) {
        char ch = (text != nullptr && *text != '\0') ? *text++ : ' ';
        oled_char(col, row, ch);
    }
}

void oled_flush() {
    if (!g_oled_ok) {
        return;
    }
    oled_cmd(0x21);
    oled_cmd(0x00);
    oled_cmd(0x7F);
    oled_cmd(0x22);
    oled_cmd(0x00);
    oled_cmd(0x07);
    for (size_t i = 0; i < sizeof(g_oled_buf); i += 16) {
        Wire.beginTransmission(g_oled_addr);
        Wire.write(0x40);
        Wire.write(&g_oled_buf[i], 16);
        Wire.endTransmission();
    }
}

bool oled_init() {
    Wire.begin(OLED_SDA, OLED_SCL, 400000);
    g_oled_addr = OLED_ADDR;
    if (!oled_probe(g_oled_addr)) {
        g_oled_addr = 0x3D;
        if (!oled_probe(g_oled_addr)) {
            return false;
        }
    }
    static const uint8_t init[] = {
        0xAE, 0xD5, 0x80, 0xA8, 0x3F, 0xD3, 0x00, 0x40,
        0x8D, 0x14, 0x20, 0x00, 0xA1, 0xC8, 0xDA, 0x12,
        0x81, 0xCF, 0xD9, 0xF1, 0xDB, 0x40, 0xA4, 0xA6,
        0x2E, 0xAF
    };
    oled_cmds(init, sizeof(init));
    g_oled_ok = true;
    oled_clear();
    oled_flush();
    return true;
}

void oled_set_header(const char* text) {
    snprintf(g_oled_header, sizeof(g_oled_header), "%s", text);
}

void oled_reset_text() {
    g_oled_text_len = 0;
    g_oled_text[0] = '\0';
}

void oled_append(const char* piece, size_t len) {
    if (!g_oled_ok || piece == nullptr || len == 0) {
        return;
    }
    if (g_oled_text_len + len + 1 > OLED_TEXT_CAP) {
        size_t drop = g_oled_text_len / 2;
        memmove(g_oled_text, g_oled_text + drop, g_oled_text_len - drop + 1);
        g_oled_text_len -= drop;
    }
    for (size_t i = 0; i < len && g_oled_text_len + 1 < OLED_TEXT_CAP; i++) {
        unsigned char c = (unsigned char)piece[i];
        if (c == '\n' || (c >= 32 && c < 127)) {
            g_oled_text[g_oled_text_len++] = (char)c;
        }
    }
    g_oled_text[g_oled_text_len] = '\0';
}

void oled_add_line(const char* text, int len) {
    if (g_oled_nlines >= OLED_MAX_LINES) {
        memmove(g_oled_lines[0], g_oled_lines[1],
                (OLED_MAX_LINES - 1) * (OLED_COLS + 1));
        g_oled_nlines = OLED_MAX_LINES - 1;
    }
    if (len < 0) {
        len = 0;
    }
    if (len > OLED_COLS) {
        len = OLED_COLS;
    }
    memcpy(g_oled_lines[g_oled_nlines], text, (size_t)len);
    g_oled_lines[g_oled_nlines][len] = '\0';
    g_oled_nlines++;
}

void oled_wrap(const char* text) {
    g_oled_nlines = 0;
    char line[OLED_COLS + 1];
    int len = 0;
    const char* p = text;
    while (*p != '\0') {
        if (*p == '\n') {
            oled_add_line(line, len);
            len = 0;
            p++;
            continue;
        }
        if (*p == ' ') {
            if (len > 0 && len < OLED_COLS) {
                line[len++] = ' ';
            }
            p++;
            continue;
        }
        const char* q = p;
        while (*q != '\0' && *q != ' ' && *q != '\n') {
            q++;
        }
        int word_len = (int)(q - p);
        if (len > 0 && len + word_len > OLED_COLS) {
            while (len > 0 && line[len - 1] == ' ') {
                len--;
            }
            oled_add_line(line, len);
            len = 0;
        }
        if (word_len > OLED_COLS) {
            for (int i = 0; i < word_len; i++) {
                if (len >= OLED_COLS) {
                    oled_add_line(line, len);
                    len = 0;
                }
                line[len++] = p[i];
            }
        } else {
            if (len + word_len > OLED_COLS) {
                oled_add_line(line, len);
                len = 0;
            }
            memcpy(line + len, p, (size_t)word_len);
            len += word_len;
        }
        p = q;
    }
    if (len > 0 || g_oled_nlines == 0) {
        oled_add_line(line, len);
    }
}

void oled_render() {
    if (!g_oled_ok) {
        return;
    }
    oled_clear();
    oled_line(0, g_oled_header, OLED_COLS);
    oled_wrap(g_oled_text);
    int start = g_oled_nlines > OLED_TEXT_ROWS ? g_oled_nlines - OLED_TEXT_ROWS : 0;
    for (int i = start; i < g_oled_nlines; i++) {
        oled_line(1 + i - start, g_oled_lines[i], OLED_COLS);
    }
    oled_flush();
}

[[noreturn]] void die(const char* msg) {
    Serial.printf("\n[fatal] %s\n", msg);
    if (g_oled_ok) {
        oled_set_header("fatal");
        oled_reset_text();
        oled_append(msg, strlen(msg));
        oled_render();
    }
    while (true) {
        delay(1000);
    }
}

void* ps_alloc(size_t n, void*) {
    void* p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (p == nullptr) {
        p = heap_caps_malloc(n, MALLOC_CAP_8BIT);
    }
    if (p == nullptr) {
        Serial.printf("[fatal] out of memory (%u bytes)\n", (unsigned)n);
        die("allocation failed");
    }
    return p;
}

void ps_free(void* p, void*) {
    heap_caps_free(p);
}

uint8_t* read_file(const char* path, size_t* out_size) {
    File f = LittleFS.open(path, "r");
    if (!f) {
        Serial.printf("[fatal] cannot open %s (run 'pio run -t uploadfs')\n", path);
        die("open failed");
    }
    size_t n = f.size();
    uint8_t* buf = (uint8_t*)ps_alloc(n, nullptr);
    size_t got = 0;
    while (got < n) {
        size_t chunk = f.read(buf + got, n - got);
        if (chunk == 0) {
            break;
        }
        got += chunk;
    }
    f.close();
    if (got != n) {
        Serial.printf("[fatal] short read on %s (%u/%u)\n", path, (unsigned)got, (unsigned)n);
        die("read failed");
    }
    *out_size = n;
    return buf;
}

void safe_print(const char* piece, size_t len) {
    if (len == 0) {
        return;
    }
    if (len == 1) {
        unsigned char byte_val = (unsigned char)piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    Serial.write((const uint8_t*)piece, len);
}

void read_line(char* buf, size_t cap) {
    size_t len = 0;
    while (true) {
        while (Serial.available() > 0) {
            int ch = Serial.read();
            if (ch == '\r') {
                continue;
            }
            if (ch == '\n') {
                buf[len] = '\0';
                return;
            }
            if (len + 1 < cap) {
                buf[len++] = (char)ch;
            }
        }
        delay(10);
    }
}

void start_generation(const char* text) {
    snprintf(g_prompt, sizeof(g_prompt), "User: %s Assistant:", text);
    ellm_status st = ellm_begin(g_model, g_prompt, 0);
    if (st != ELLM_OK) {
        const char* err = ellm_status_str(st);
        Serial.printf("[error] %s\n", err);
        if (g_oled_ok) {
            oled_set_header("error");
            oled_reset_text();
            oled_append(err, strlen(err));
            oled_render();
        }
        return;
    }
    Serial.print(g_prompt);
    if (g_oled_ok) {
        oled_set_header("generating");
        oled_reset_text();
        oled_append("> ", 2);
        oled_append(text, strlen(text));
        oled_append("\n", 1);
        oled_render();
    }
    g_generating = true;
    g_generated = 0;
    g_start_ms = 0;
}

void finish_generation() {
    Serial.println();
    int tps = 0;
    if (g_generated > 0 && g_start_ms != 0) {
        uint32_t elapsed = millis() - g_start_ms;
        float rate = elapsed ? (g_generated * 1000.0f / elapsed) : 0.0f;
        tps = (int)(rate + 0.5f);
        Serial.printf("[%d tokens, %.1f tok/s]\n", g_generated, rate);
    }
    if (g_oled_ok) {
        if (g_generated > 0) {
            snprintf(g_oled_header, sizeof(g_oled_header), "done %d tok/s", tps);
        } else {
            oled_set_header("ready");
        }
        oled_render();
    }
    g_generating = false;
}

}  // namespace

void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.println();
    Serial.println("esp32-llm demo");

    g_oled_ok = oled_init();
    if (g_oled_ok) {
        oled_set_header("esp32-llm");
        oled_reset_text();
        oled_append("loading model...", 16);
        oled_render();
        Serial.println("OLED: SSD1306 128x64");
    } else {
        Serial.println("[warn] OLED not found on I2C (SDA=8, SCL=9)");
    }

    Serial.printf("PSRAM: %u bytes total, %u free\n", (unsigned)ESP.getPsramSize(), (unsigned)ESP.getFreePsram());

    if (!psramFound() || ESP.getPsramSize() == 0) {
        die("PSRAM not detected; select an N16R8 board with qio_opi memory type");
    }
    if (!LittleFS.begin(false, "/littlefs", 10, "spiffs")) {
        die("LittleFS mount failed; run 'pio run -t uploadfs' first");
    }

    Serial.println("loading model...");
    size_t model_size = 0;
    size_t tokenizer_size = 0;
    uint8_t* model_data = read_file("/model_int8.bin", &model_size);
    uint8_t* tokenizer_data = read_file("/tokenizer.bin", &tokenizer_size);

    ellm_allocator allocator;
    allocator.alloc = ps_alloc;
    allocator.free = ps_free;
    allocator.ctx = nullptr;

    ellm_options options;
    memset(&options, 0, sizeof(options));
    options.allocator = &allocator;
    options.dequantize_embeddings = 1;
    options.temperature = 0.7f;
    options.top_p = 0.9f;
    options.seed = ((uint64_t)esp_random() << 32) | esp_random();
    if (options.seed == 0) {
        options.seed = 1;
    }

    ellm_status st = ellm_load(model_data, model_size, tokenizer_data, tokenizer_size,
                               &options, &g_model);
    if (st != ELLM_OK) {
        Serial.printf("[fatal] %s\n", ellm_status_str(st));
        die("model load failed");
    }

    g_piece_cap = ellm_max_token_length(g_model) + 1;
    g_piece = (char*)ps_alloc(g_piece_cap, nullptr);

    if (g_oled_ok) {
        oled_set_header("ready");
        oled_reset_text();
        oled_append("type on serial", 15);
        oled_render();
    }

    Serial.printf("model: vocab=%d ctx=%d\n", ellm_vocab_size(g_model), ellm_context_size(g_model));
    Serial.printf("ready; PSRAM free: %u bytes\n", (unsigned)ESP.getFreePsram());
    Serial.println("type a message and press enter");
}

void loop() {
    if (!g_generating) {
        static char line[MAX_LINE];
        Serial.print("\nUser: ");
        read_line(line, sizeof(line));
        if (line[0] != '\0') {
            start_generation(line);
        }
        return;
    }

    size_t len = 0;
    ellm_step step = ellm_next(g_model, g_piece, g_piece_cap, &len);
    if (step == ELLM_STEP_TOKEN) {
        if (g_generated == 0) {
            g_start_ms = millis();
        }
        safe_print(g_piece, len);
        if (g_oled_ok) {
            oled_append(g_piece, len);
            oled_render();
        }
        g_generated++;
        return;
    }
    if (step == ELLM_STEP_PROMPT) {
        return;
    }
    if (step == ELLM_STEP_ERROR) {
        Serial.println("\n[error] generation failed");
        if (g_oled_ok) {
            oled_set_header("error");
            oled_append("\ngeneration failed", sizeof("\ngeneration failed") - 1);
            oled_render();
        }
        g_generating = false;
        return;
    }
    finish_generation();
}

import random
import sys

FACTS = [
    ("What is quantization?", "Quantization stores weights with fewer bits, such as int8 instead of float32, so the model is smaller."),
    ("What is an ESP32?", "The ESP32 is a low cost microcontroller with Wi-Fi and Bluetooth."),
    ("What is PSRAM?", "PSRAM is an external RAM chip that gives the ESP32 more memory than its internal SRAM."),
    ("What is flash memory?", "Flash memory keeps your program and model files even when the power is off."),
    ("What is a token?", "A token is a small piece of text, like a word or part of a word, that the model reads."),
    ("What is a tokenizer?", "A tokenizer splits text into tokens and turns them into numbers."),
    ("What is a transformer?", "A transformer is a neural network that uses attention to decide which earlier tokens matter."),
    ("What is attention?", "Attention lets each token look at earlier tokens and weigh how important they are."),
    ("What is TinyML?", "TinyML is machine learning that runs on very small, low power devices."),
    ("What is inference?", "Inference means using a trained model to make predictions."),
    ("What is training?", "Training adjusts the weights of a model so its predictions get better."),
    ("What is a weight?", "A weight is a number the model learns during training."),
    ("What is int8?", "Int8 is a number format that uses one byte and stores whole numbers from minus 128 to 127."),
    ("How many bytes is a float32?", "A float32 uses four bytes."),
    ("How many bytes is an int8?", "An int8 uses one byte."),
    ("What is a KV cache?", "A KV cache stores the keys and values of earlier tokens so they are not recomputed."),
    ("What is loss?", "Loss is a number that says how wrong the model is. Lower is better."),
    ("What is overfitting?", "Overfitting is when a model memorizes its training data and does badly on new data."),
    ("What is a microcontroller?", "A microcontroller is a tiny computer on one chip that runs a single program."),
    ("What limits tokens per second on an ESP32?", "Mostly memory bandwidth, because every weight is read for every token."),
    ("Hi", "Hello! Ask me about TinyML."),
    ("Who are you?", "I am a tiny language model running on a microcontroller."),
]
PREFIX = ["", "Hi! ", "Quick question. ", "Hey, "]

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "toy_corpus.txt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    rng = random.Random(0)
    with open(out, "w") as f:
        for _ in range(n):
            q, a = rng.choice(FACTS)
            f.write(f"User: {rng.choice(PREFIX)}{q} Assistant: {a}\n")
    print(f"wrote {n} lines to {out}")

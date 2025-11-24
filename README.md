# ☠️ Neural-C: The Unga Bunga Deep Learning Framework

> "Libraries? Where we're going, we don't need libraries." - *Captain Espadara*

**Neural-C** is a raw, bare-metal Machine Learning framework written in ANSI C. While the rest of the world is drowning in Python dependencies and 4GB Conda environments, this project fits in a single file and runs on a toaster.

We don't use TensorFlow. We use **Math**.

## ⚓ Features

* **Zero Dependencies:** No Python, no NumPy, no Torch. Just `stdlib.h` and pure grit.
* **Hand-Rolled Matrix Engine:** Because `malloc` is the only abstraction you need.
* **Backpropagation from Scratch:** We calculate derivatives manually like our ancestors intended.
* **32-Neuron Hidden Layer:** Overkill for XOR? Yes. Do we care? No.
* **Memory Safe:** Now with `memset` because pirates don't like leaks.
* **Pirate Certified:** Header included.

## 🏴‍☠️ The Evolution

This project documents the journey from "Unga Bunga" logic to Industry Standard Backpropagation:

1.  **Level 1: The Guesser.** Randomly wiggling weights to see if the error goes down (Finite Difference).
2.  **Level 2: The Architect.** Building a Matrix engine to handle layers dynamically.
3.  **Level 3: The Mathematician.** Implementing the Chain Rule (Backpropagation) for instant gradient calculation.

## ⚔️ Usage

Compile it like a pirate (with flags).

```bash
gcc -Wall -Wextra -O3 xor_32.c -o neural_c -lm
./neural_c

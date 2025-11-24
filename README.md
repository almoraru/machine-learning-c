# ☠️ Neural-C: The Caveman's Learning Framework

> "Libraries? Where we're going, we don't need libraries." - *Captain Espadara*

**Neural-C** is a raw, bare-metal Machine Learning framework written in ANSI C. While the rest of the world is drowning in Python dependencies and 4GB Conda environments, this project fits in a single file, compiles instantly, and runs faster than a cannonball.

**Current Status:**

  * **Neurons:** 64 Hidden Units (Overkill mode).
  * **Iterations:** 100,000+ training loops in milliseconds.
  * **Dependencies:** `0`.

## ⚓ Features

  * **Zero Dependencies:** No Python, no NumPy, no Torch. Just `stdlib.h` and pure grit.
  * **Hand-Rolled Matrix Engine:** We flatten 2D matrices into 1D memory for maximum cache efficiency.
  * **Backpropagation:** We implemented the Chain Rule from scratch. No "finite difference" guessing.
  * **Memory Safe:** Uses `memset` and strictly managed pointers. No leaks on this ship.
  * **Pirate Certified:** Header included.

## 🔧 Under the Hood: Why is it so fast?

Most "beginner" neural networks use **Finite Difference** to learn. They wiggle a weight, check the error, and wiggle it back. If you have 1,000 weights, you have to run the model 1,000 times just to take **one** step.

**Neural-C uses Backpropagation.**

Instead of guessing, we use Calculus. By applying the **Chain Rule**, we can calculate the perfect direction for *every single weight* in the network in just **one** backward pass.

1.  **Forward Pass:** Data flows input $\to$ output.
2.  **Backward Pass:** Error flows output $\to$ input.
3.  **Update:** We nudge all 64 neurons simultaneously.

This changes the time complexity from $O(N^2)$ to $O(N)$. That is why we can run 100,000 iterations in the blink of an eye.

## 🏴‍☠️ The Architecture

The current model `xor_32.c` (configurable to 64) is a Multi-Layer Perceptron (MLP).

```
[Input Layer]       [Hidden Layer]       [Output Layer]
(2 neurons)    -->  (64 neurons)    -->  (1 neuron)
   x1, x2           "The Black Box"        Result
```

### The Matrix Engine

We treat flat arrays as 2D matrices using `row * stride + col`. This avoids the overhead of pointers-to-pointers.

```c
// The beating heart of the system
#define MAT_AT(m, i, j) (m).es[(i)*(m).cols + (j)]
```

## ⚔️ Usage

Compile it like a pirate (with optimization flags).

```bash
gcc -Wall -Wextra -O3 xor_32.c -o neural_c -lm
./neural_c
```

## 📜 The Evolution

This project documents the journey from "Caveman's" logic to Industry Standard:

1.  **Level 1: The Guesser.** Randomly wiggling weights (Finite Difference).
2.  **Level 2: The Architect.** Building a generic Matrix engine.
3.  **Level 3: The Mathematician.** Implementing Backpropagation.

## ⚠️ Warning

May contain traces of pointer arithmetic and manual memory management. If you forget to `free()`, the ship sinks.

-----

*By espadara [espadara@pirate.capn.gg](mailto:amoraru.dev@gmail.com)*

-----


# deep-learning-lib-with-python

🇺🇸 English | 🇧🇷 [Português](README.md)

> A feedforward neural network (MLP) implemented from scratch in Python, with manual backpropagation and topology visualization.

## Overview

This project implements a **multilayer perceptron (MLP) from scratch**, using only **NumPy** for linear algebra. It depends on no deep-learning framework (TensorFlow, PyTorch) — the forward pass, backpropagation, and weight updates are written by hand via the chain rule. It is a didactic resource for understanding what happens "under the hood" of a deep-learning library.

The included example trains the network on the classic **XOR** problem.

## What's implemented

The `NeuralNetwork` class (in `RedeNeural.py`) provides:

- **`__init__(neurons_input, neurons_hidden_layers, neurons_output)`** — configurable architecture with an arbitrary number of hidden layers; weights and biases randomly initialized.
- **`feedforward(input)`** — forward propagation with **sigmoid** activation in every layer.
- **`backprop(input, y)`** — manual backpropagation via the chain rule, computing weight and bias gradients layer by layer.
- **`train(x, y, num_iterations)`** — training loop (full-batch gradient descent).
- **`visualize_nn(nn)`** — draws the network topology as a directed graph using **NetworkX** + **Matplotlib**.

Helper functions: `sigmoid` and `sigmoid_derivative`.

## Running

```bash
pip install numpy networkx matplotlib
python RedeNeural.py
```

The script:
1. Trains the network on the XOR dataset for 1500 iterations.
2. Opens a Matplotlib window with the network topology.
3. Enters an interactive loop where you type comma-separated input values (e.g. `1,0,1`) and see the predicted output. Type `q` to quit.

## Project status

Functional educational implementation in **a single file**. Honest caveats:

- **Not yet an importable library** — the training code, the XOR dataset, and the interactive loop run at module level, so importing `RedeNeural` triggers training. To become a reusable library, that example code should move into an `if __name__ == "__main__":` block.
- Only sigmoid activation and an implicit squared loss (no choice of loss/activation).
- No mini-batches, configurable learning rate, regularization, or early stopping.
- No automated tests.

## Suggested roadmap

- [ ] Wrap the example in `if __name__ == "__main__":` and expose the API as a package.
- [ ] Make the learning rate a parameter.
- [ ] Add ReLU/tanh and a choice of loss function.
- [ ] Add `requirements.txt` and `examples/`.

## License

This project does not yet declare a license. Until one is added, all rights are reserved by the author.

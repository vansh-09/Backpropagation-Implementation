
# Backpropagation from Scratch (XOR Problem)

A clean, from-scratch replication of the 1986 backpropagation algorithm using only NumPy to solve the XOR problem.

---

## Live Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

---

## Table of Contents
- Overview
- Architecture
- Forward Pass
- Loss Function
- Backpropagation Math
- Training Loop Explained
- Final Predictions
- Visualizations
- Learning Outcomes

---

## Overview

This project is a step-by-step implementation of the backpropagation algorithm to train a neural network that learns to model the XOR logic gate. The XOR problem is historically important because it proved that linear models weren’t enough and required deeper architectures.

---

## Architecture
```
Input Layer      : 2 neurons
Hidden Layer     : 4 neurons
Output Layer     : 1 neuron
Activation Func  : Sigmoid
Loss Function    : Mean Squared Error (MSE)
```

---

## Forward Pass
```
Step 1: Compute input to hidden layer
        h_input = X • W1
Step 2: Apply activation
        h_output = sigmoid(h_input)
Step 3: Compute input to output layer
        o_input = h_output • W2
Step 4: Apply activation
        o_output = sigmoid(o_input)
```

---

## Loss Function

We use Mean Squared Error:

\[ \text{Loss} = \frac{1}{n} \sum (y - \hat{y})^2 \]

Where:
- \( y \) = target output
- \( \hat{y} \) = predicted output
- \( n \) = number of samples

---

## Backpropagation Math (Annotated)

### Output Layer Delta:
\[ \delta_j = o_j (1 - o_j) (t_j - o_j) \]

### Hidden Layer Delta:
\[ \delta_j = o_j (1 - o_j) \sum_k w_{jk} \delta_k \]

### Weight Update Rule:
\[ W = W + \eta \cdot \delta \cdot \text{input}^T \]

---

## Training Loop Explained
```python
for epoch in range(20000):
    # forward pass
    # compute error
    # backward pass
    # update weights
```
- Weights are updated using gradient descent with sigmoid derivatives
- Learning rate: 0.3
- Hidden size: 4 neurons

---

## Final Predictions (Rounded Output)
```
[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0
```

---

## Visualizations
- Loss curve showing convergence
- Weight evolution graph
- Backpropagation flowchart

---

## Learning Outcomes
- Understand neural computation step-by-step
- Learn to apply gradients manually using math
- See the full backpropagation cycle in action
- Reproduce original neural net papers with confidence

---

# Perceptron from Scratch (AND Gate)

A foundational neural network implementation based on the classic Perceptron model that learns to classify binary AND logic.

---

## Live Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

---

## What is a Perceptron?
A perceptron is the most basic building block of a neural network. It takes multiple binary inputs, multiplies them with weights, adds a bias, and passes them through a step function.

---

## Model Description
```
Inputs      : 2 (binary: 0 or 1)
Output      : 1 (either 0 or 1)
Activation  : Step function
Training    : Perceptron Learning Rule
```

---

## Perceptron Learning Rule

\[ y = f(w \cdot x + b) \]
\[ \text{if } y \neq target:\quad w = w + \eta (target - y) x \]

---

## Training Data (AND Gate)
```python
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

Y = [0, 0, 0, 1]
```

---

## Code Breakdown
- Weight initialization
- Prediction using step function
- Training using simple update rule

---

## Final Output
```
Prediction after training:
[0, 0] → 0
[0, 1] → 0
[1, 0] → 0
[1, 1] → 1
```

---

## Key Learning Points
- Understand how logic gates can be learned
- How bias allows the perceptron to shift decision boundaries
- What the limitation of a single-layer perceptron is (cannot solve XOR)

---

Want to learn more? Try extending this to OR / NAND gates or multi-layer perceptrons.

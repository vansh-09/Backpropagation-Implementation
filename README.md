
# Backpropagation from Scratch (XOR Problem)

A clean, from-scratch replication of the 1986 backpropagation algorithm using only NumPy to solve the XOR problem.

---

## Live Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WGahZt6WeafM1Wxb80wfVjK5ZB_tnXLU)

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
$$ \[ \delta_j = o_j (1 - o_j) (t_j - o_j) \] $$

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

Want to learn more? Try extending this to OR / NAND gates or multi-layer perceptrons.

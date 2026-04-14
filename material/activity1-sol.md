<!-- 
# Activity 1: From Perceptrons to Simple Neural Networks

## Introduction

In this lab, we'll bring the concepts from our lecture notes to life. We'll start by exploring the **Perceptron**, the simplest building block, and see what it can (and can't!) do. Then, we'll build our very first **Multi-Layer Perceptron (MLP)** using a popular library called Keras to tackle a problem the single Perceptron couldn't solve.

**Learning Objectives:**

1.  Implement a Perceptron function in Python.
2.  Understand how weights and bias affect a Perceptron's output for logic gates (AND, OR, NOT).
3.  Visualize and explain why a single Perceptron fails on the non-linearly separable XOR problem.
4.  Build a simple MLP with one hidden layer using Keras.
5.  Understand the role of layers and activation functions (like ReLU and Sigmoid) in an MLP.
6.  Train and evaluate a basic neural network to solve the XOR problem.

---

## Part 1: The Perceptron - A Single Neuron's Power and Limits

Remember from the lecture, the Perceptron is an early model of an artificial neuron. It takes inputs, multiplies them by weights, adds a bias, and uses a simple *step function* to decide whether to output 1 or 0.

### 1.1 Implementing the Perceptron

Let's write a Python function for our Perceptron. We'll use NumPy for efficient calculations.

```python
import numpy as np
import matplotlib.pyplot as plt

def perceptron(inputs, weights, bias):
    """Calculates the output of a simple perceptron with a step activation function."""
    # Ensure inputs and weights match in number
    # Note: np.dot performs the element-wise multiplication and sum!
    weighted_sum = np.dot(inputs, weights) + bias

    # --- YOUR CODE HERE --- #
    # Apply the step activation function:
    # If weighted_sum is greater than or equal to 0, output 1
    # Otherwise, output 0
    if weighted_sum >= 0:
        # What should the function return if it "fires"?
        output = 1
    else:
        # What should the function return if it doesn't "fire"?
        output = 0
    # --- END YOUR CODE --- #

    return output

# Let's test it with some dummy values
test_inputs = [1, 0]
test_weights = [0.5, 0.5]
test_bias = -0.7
print(f"Test Input: {test_inputs}")
print(f"Test Output: {perceptron(test_inputs, test_weights, test_bias)}") # Should be 0 (0.5*1 + 0.5*0 + (-0.7) = -0.2 < 0)

test_inputs = [1, 1]
print(f"\nTest Input: {test_inputs}")
print(f"Test Output: {perceptron(test_inputs, test_weights, test_bias)}") # Should be 1 (0.5*1 + 0.5*1 + (-0.7) = 0.3 >= 0)
```

**Task:** Fill in the `YOUR CODE HERE` section in the `perceptron` function above to implement the step activation logic. Run the cell to test your implementation.

### 1.2 Modeling Logic Gates

Now, let's see if we can make our Perceptron act like basic logic gates by choosing the right `weights` and `bias`.

**Inputs for 2-input gates:**
`gate_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]`

**(a) OR Gate:** Outputs 1 if *any* input is 1.

```python
gate_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

# --- YOUR TASK: Find suitable weights and bias for OR --- #
# Hint: We want the weighted sum to be >= 0 when at least one input is 1.
weights_or = [1.0, 1.0] # Possible weights
bias_or = -0.5     # Possible bias (needs to be slightly negative)
# --- END YOUR TASK --- #

print("--- OR Gate ---")
for input_pair in gate_inputs:
    output = perceptron(input_pair, weights_or, bias_or)
    print(f"Input: {input_pair}, Output: {output}")
```

**Task:** Adjust `weights_or` and `bias_or` in the cell above until the Perceptron correctly simulates the OR gate. Run the cell to check your results. (The suggested values should work!).

**(b) AND Gate:** Outputs 1 only if *both* inputs are 1.

```python
# --- YOUR TASK: Find suitable weights and bias for AND --- #
# Hint: We need a higher threshold this time. Only (1,1) should activate it.
weights_and = [1.0, 1.0] # Possible weights
bias_and = -1.5    # Possible bias (needs to be more negative than OR)
# --- END YOUR TASK --- #

print("\n--- AND Gate ---")
for input_pair in gate_inputs:
    output = perceptron(input_pair, weights_and, bias_and)
    print(f"Input: {input_pair}, Output: {output}")
```

**Task:** Adjust `weights_and` and `bias_and` in the cell above until the Perceptron correctly simulates the AND gate. Run the cell to check.

**(c) NOT Gate:** (Single input) Outputs the opposite of the input.

```python
# Input for 1-input gate:
not_inputs = [[0], [1]] # Note: inputs are lists containing single elements

# --- YOUR TASK: Find suitable weight and bias for NOT --- #
# Hint: The weight should probably be negative.
weights_not = [-1.0] # Possible weight
bias_not = 0.5     # Possible bias
# --- END YOUR TASK --- #

print("\n--- NOT Gate ---")
for input_val in not_inputs:
     # Make sure weights_not matches the input structure (single element)
    output = perceptron(input_val, weights_not, bias_not)
    print(f"Input: {input_val}, Output: {output}")
```

**Task:** Adjust `weights_not` and `bias_not` for the NOT gate.

**Reflection:** You've just shown how changing weights and biases allows a simple Perceptron to perform different linear computations!

### 1.3 The Wall: Linear Separability and XOR

As discussed in the lecture, the Perceptron works by finding a *linear* boundary (a straight line in 2D) to separate the data points. Let's visualize this for AND and XOR.

```python
# Define the input points
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
x_coords = inputs[:, 0]
y_coords = inputs[:, 1]

# Define the outputs for AND and XOR
outputs_and = np.array([0, 0, 0, 1])
outputs_xor = np.array([0, 1, 1, 0])

# Define colors (blue for 0, red for 1)
colors = ['blue', 'red']

# Function to create a plot (from lecture notes)
def plot_gate(gate_outputs, title):
    plt.figure(figsize=(5, 5)) # Smaller size for lab
    plt.title(title)
    plt.xlabel('Input 1'); plt.ylabel('Input 2')
    for i in range(len(inputs)):
        plt.scatter(x_coords[i], y_coords[i], color=colors[gate_outputs[i]], s=150, zorder=3)
    plt.xticks([0, 1]); plt.yticks([0, 1])
    plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)
    plt.grid(True, zorder=0); plt.axhline(0, color='black', linewidth=0.5); plt.axvline(0, color='black', linewidth=0.5)
    legend_handles = [plt.scatter([], [], color='blue', s=50, label='Output 0'),
                      plt.scatter([], [], color='red', s=50, label='Output 1')]
    plt.legend(handles=legend_handles, loc='best') # Use 'best' location
    plt.show()

# Plot AND and XOR
print("--- Visualizing Logic Gates ---")
plot_gate(outputs_and, 'AND Gate Outputs')
plot_gate(outputs_xor, 'XOR Gate Outputs')

```

**Observe:**
*   For the **AND** gate, can you imagine drawing a straight line to separate the blue dots (output 0) from the red dot (output 1)? Yes! This is **linearly separable**.
*   For the **XOR** gate, try to imagine drawing *one single straight line* that separates the blue dots from the red dots. It's impossible! This is **non-linearly separable**.

**The XOR Challenge:**

```python
# --- YOUR TASK: Try to find weights/bias for XOR --- #
# See if you can find weights_xor and bias_xor that work for all 4 inputs
weights_xor = [?, ?] # Try different values
bias_xor = ?       # Try different values
# --- END YOUR TASK --- #

print("\n--- XOR Gate Challenge ---")
correct_count = 0
for i, input_pair in enumerate(gate_inputs):
    output = perceptron(input_pair, weights_xor, bias_xor)
    expected = outputs_xor[i]
    is_correct = (output == expected)
    print(f"Input: {input_pair}, Expected: {expected}, Got: {output} -> {'Correct' if is_correct else 'WRONG'}")
    if is_correct:
        correct_count += 1

print(f"\nTotal Correct: {correct_count} out of 4")
if correct_count < 4:
    print("It seems the single Perceptron struggles with XOR!")
else:
    print("Wow! Are you sure? Double-check your weights/bias!")

```

**Task:** Try to find `weights_xor` and `bias_xor` that make the `perceptron` function work correctly for the XOR gate. Run the cell multiple times with different values.

**Question:** Were you able to find weights and a bias for the single Perceptron to solve XOR correctly for all four inputs? Explain why or why not, based on the concept of linear separability discussed in the lecture and the visualization above.

**(Write your explanation in a text cell below)**

> *answer placeholder: e.g., No, it's impossible because the XOR data points cannot be separated by a single straight line (they are not linearly separable). A single perceptron can only create a linear decision boundary.*

**Part 1 Conclusion:** We've seen that a single Perceptron is like a linear classifier. It's useful for simple tasks but fails when the data isn't linearly separable, like the XOR problem. To solve more complex problems, we need more power!

---

## Part 2: Multi-Layer Networks with Keras - Solving XOR

To handle non-linear problems, we need networks with multiple layers (specifically, hidden layers) and neurons that use **non-linear activation functions** (like ReLU or Sigmoid), as explained in the lecture. We'll use **Keras**, a user-friendly library (part of TensorFlow), to build such a network – an MLP.

### 2.1 Setup and Data Preparation

First, let's import Keras and prepare our XOR data in a format Keras understands.

```python
# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # We'll use this to define layers

print("TensorFlow version:", tf.__version__)

# XOR input data (already defined, but let's make sure it's a NumPy array)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

# XOR output data (needs to be NumPy array, and often needs reshaping for Keras)
y = np.array([[0], [1], [1], [0]], dtype=np.float32) # Output 0 or 1

print("\nX (Inputs):\n", X)
print("\ny (Expected Outputs):\n", y)
```

### 2.2 Building the MLP Model

We will build a simple MLP with:
1.  An **Input Layer** (implicitly defined by the first `Dense` layer's `input_shape`).
2.  One **Hidden Layer** with a few neurons (e.g., 4) and a non-linear activation function (ReLU).
3.  An **Output Layer** with one neuron (outputting 0 or 1) and a Sigmoid activation function (suitable for binary classification).

```python
# --- Define the Model Structure ---

model = keras.Sequential(name="Simple_MLP_for_XOR") # Give our model a name

# Input Layer: Keras figures out the input size from 'input_shape' in the first layer.
# Our input has 2 features (Input 1, Input 2).

# --- YOUR CODE HERE --- #
# Add the Hidden Layer:
# Use layers.Dense(). We need to specify:
# - units: How many neurons in this layer? Let's start with 4.
# - activation: Which non-linear function? Let's use 'relu'.
# - input_shape: Only needed for the *first* layer. It's a tuple (num_features,). Our input has 2 features.
model.add(layers.Dense(units=4, activation='relu', input_shape=(2,)))

# Add the Output Layer:
# Use layers.Dense(). We need:
# - units: How many output values? Just 1 (0 or 1).
# - activation: For binary (0/1) output, 'sigmoid' is a common choice.
#   (Sigmoid squashes output between 0 and 1, like a probability)
model.add(layers.Dense(units=1, activation='sigmoid'))
# --- END YOUR CODE --- #


# Print a summary of the model's layers and parameters (weights/biases)
model.summary()
```

**Task:** Fill in the `YOUR CODE HERE` section to add the hidden and output layers using `layers.Dense()`. Check the comments for guidance on the parameters (`units`, `activation`, `input_shape`). Run the cell to see the model summary.

**Question:** Look at the `model.summary()`. How many parameters (weights + biases) does the hidden layer have? How many does the output layer have? Can you figure out how these numbers are calculated based on the number of inputs and neurons in each layer?

> *Student answer placeholder: Hidden layer: (2 inputs * 4 neurons) + 4 biases = 12 params. Output layer: (4 inputs from hidden layer * 1 neuron) + 1 bias = 5 params. Total = 17.*

### 2.3 Compiling the Model

Before training, we need to "compile" the model. This configures it for learning by specifying:
*   **Optimizer:** The algorithm used to update weights (like Gradient Descent). `adam` is a popular and effective choice.
*   **Loss Function:** How we measure the error between the network's predictions and the true outputs. `binary_crossentropy` is standard for binary (0/1) classification problems.
*   **Metrics:** What metric(s) to track during training. `accuracy` is common for classification.

```python
# --- Compile the model ---
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Model compiled successfully!")
```

**Task:** Run the cell above to compile the model. No code changes needed here, just understanding the components.

### 2.4 Training the Model

Now, the exciting part – training! We feed the model our XOR data (`X`, `y`) and ask it to learn using the `fit` method. Keras handles the Forward and Backward passes automatically.

*   **`epochs`:** How many times the model sees the entire dataset.
*   **`batch_size`:** How many samples to process before updating weights (we can use the whole small dataset here, so `batch_size=4`).
*   **`verbose`:** How much information to print during training (1 shows a progress bar).

```python
# --- Train the model ---
print("Starting training...")

# --- YOUR TASK: Set the number of epochs --- #
# Training might take a few hundred or thousand epochs for XOR. Start with 500.
num_epochs = 500
# --- END YOUR TASK --- #

history = model.fit(X, y,
                    epochs=num_epochs,
                    batch_size=4, # Process all 4 samples at once
                    verbose=1) # Show progress

print("Training finished!")

# You can optionally plot the training loss and accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()
```

**Task:** Choose a number of `epochs` (start with 500) and run the cell to train the network. Observe the `loss` decreasing and `accuracy` increasing.

### 2.5 Evaluating and Predicting

Let's see how well our trained model performs on the XOR data.

```python
# --- Evaluate the model ---
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\nEvaluation on XOR data:")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f} (or {accuracy*100:.2f}%)")

# --- Make predictions ---
predictions = model.predict(X)
print("\nModel Predictions (raw output from sigmoid):")
print(predictions)

# Let's make the predictions clearer (0 or 1) by rounding
rounded_predictions = np.round(predictions)
print("\nRounded Predictions (0 or 1):")
print(rounded_predictions)

print("\nComparison:")
print("Inputs | Expected | Predicted")
print("-------|----------|-----------")
for i in range(len(X)):
    print(f"{X[i]} | {y[i][0]}        | {int(rounded_predictions[i][0])}")

# Check if all predictions match the expected output
if np.array_equal(y, rounded_predictions):
    print("\nSuccess! The MLP learned to solve the XOR problem.")
else:
    print("\nClose! The model might need more training (epochs) or a slightly different structure.")

```

**Task:** Run the cell above to evaluate the model and see its predictions.

**Question:** Did the MLP successfully learn the XOR function (achieve 100% accuracy)? Compare this result to the single Perceptron's attempt in Part 1. Why was the MLP able to succeed where the Perceptron failed?

> *answer placeholder: Yes, the MLP likely achieved 100% accuracy. It succeeded because it has a hidden layer with non-linear ReLU activation functions, allowing it to learn a non-linear decision boundary necessary to separate the XOR data points.*

### 2.6 Your Turn (Experiment!)

Let's experiment slightly. Go back to cell in **Section 2.4 (Training the Model)**. Try changing the `num_epochs`.
*   What happens if you use only 50 epochs? Re-run the training and evaluation cells. Does it still learn XOR perfectly?
*   What happens if you use 2000 epochs?

**(Optional Advanced):** Go back to **Section 2.2 (Building the Model)**. Try changing the number of `units` in the hidden layer (e.g., to 2 or 8). Re-run the build, compile, train, and evaluate cells. Does it still work? Does it train faster or slower?

**(Record your observations in a text cell below)**

> *answer placeholder: e.g., With only 50 epochs, the accuracy might not reach 100% because the model didn't have enough time to learn. With 2000 epochs, it should still work well. Changing hidden units might affect how quickly it learns, but 2-8 units should still be able to solve XOR.*

### 2.7 Troubleshooting and Further Investigation (Optional)

Sometimes, even with the right structure, a neural network might not get perfect results on the first try, especially with random starting weights. If your MLP didn't quite reach 100% accuracy on XOR, or if you just want to explore more, here are some common things you can try changing:

**1. Train for Longer (More Epochs):**
*   **Why:** Gradient descent takes small steps. Maybe the model just needed more steps to find the best weights.
*   **How:** Go back to **Section 2.4 (Training the Model)** and increase the `num_epochs` variable (e.g., try 1000, 1500, or even 3000). Re-run the training and evaluation cells.

**2. Try Training Again (Different Random Start):**
*   **Why:** Neural networks start with random weights. Sometimes, you get an "unlucky" starting point that makes it hard for the optimizer to find the best solution.
*   **How:** Simply re-run the cells in order starting from **Section 2.2 (Building the Model)**, then **Section 2.3 (Compiling)**, **Section 2.4 (Training)**, and **Section 2.5 (Evaluating)**. A different random initialization might lead to a better result this time!

**3. Adjust Hidden Layer Size (Number of Units):**
*   **Why:** The number of neurons in the hidden layer determines the model's "capacity" or complexity. Too few might not be enough to learn XOR; sometimes more can help (up to a point).
*   **How:** Go back to **Section 2.2 (Building the Model)**. Change the `units` parameter in the *hidden* `layers.Dense()` (e.g., try `units=8`, `units=2`, or `units=16`). Remember to re-run the build, compile, train, and evaluate cells afterward.

**4. Change the Hidden Layer Activation Function:**
*   **Why:** We used `'relu'`, which is common, but other non-linear functions exist. They process the weighted sums differently.
*   **How:** In **Section 2.2**, change `activation='relu'` in the *hidden* layer to `activation='tanh'` or maybe `activation='sigmoid'`. Re-run the subsequent cells to see if it makes a difference. (Note: `'sigmoid'` in hidden layers is less common now but okay for experimentation here).

**5. Add Another Hidden Layer (Increase Depth):**
*   **Why:** Making the network "deeper" gives it more stages to transform the data, potentially allowing it to find more complex patterns. (As discussed, this might be overkill for XOR but is a common technique for harder problems).
*   **How:** Modify **Section 2.2** as shown in the example provided previously (inserting another `layers.Dense(units=..., activation=...)` line before the output layer). Remember to rebuild, recompile, retrain, and re-evaluate.

**6. Change the Optimizer (More Advanced):**
*   **Why:** `adam` is a good default, but other optimization algorithms exist (like `sgd` - Stochastic Gradient Descent, or `rmsprop`). They update weights differently.
*   **How:** In **Section 2.3 (Compiling the Model)**, change `optimizer='adam'` to `optimizer='sgd'` or `optimizer='rmsprop'`. You might need to adjust `epochs` significantly if you change the optimizer (e.g., `sgd` often requires more epochs or careful tuning of its learning rate).

**Experimentation Tips:**

*   **Change one thing at a time!** This helps you understand the effect of each specific change.
*   **Keep track of what you tried:** Maybe make notes in a text cell about the changes you made and the resulting accuracy.
*   **Don't expect huge differences on XOR:** Since XOR is relatively simple for an MLP, many of these changes might still result in 100% accuracy once trained sufficiently. The goal here is to understand *how* these components can be adjusted.



---

## Lab Conclusion

You've successfully:

*   Implemented a Perceptron and tested its ability to model simple logic gates.
*   Encountered the limits of a single Perceptron with the non-linearly separable XOR problem.
*   Built, trained, and evaluated a Multi-Layer Perceptron (MLP) using Keras.
*   Demonstrated that the MLP, with its hidden layer and non-linear activation functions, *can* solve the XOR problem.

This lab shows the fundamental step up from single neurons to layered networks, which forms the basis for much more complex Deep Learning models used today! 
-->
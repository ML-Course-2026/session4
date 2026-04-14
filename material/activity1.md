# Activity 1: From Perceptrons to Multi-Layer Neural Networks

## Introduction

This laboratory exercise operationalizes the theoretical concepts discussed in the lecture. The objective is to programmatically define a single Perceptron to evaluate its computational capacity and limitations regarding linear separability. Subsequently, the exercise utilizes the Keras framework to construct a Multi-Layer Perceptron (MLP) capable of processing non-linear datasets.

**Learning Objectives:**
1.  Implement a mathematical Perceptron function in Python.
2.  Manipulate weights and biases to model fundamental logic gates (AND, OR, NOT).
3.  Empirically and visually demonstrate the Perceptron's failure on the non-linearly separable XOR problem.
4.  Construct an MLP with a hidden layer using the Keras library.
5.  Analyze the role of hidden layers and non-linear activation functions.
6.  Train and evaluate a neural network to solve XOR and visualize non-linear decision boundaries.

---

## Part 1: The Perceptron - Computational Capacity and Limits

The Perceptron acts as a linear classifier. It computes the dot product of input vectors and weight vectors, adds a bias scalar, and applies a Heaviside step function to yield a binary output.

### 1.1 Implementing the Perceptron

The following Python function models the Perceptron utilizing NumPy for optimized vector operations.

```python
import numpy as np
import matplotlib.pyplot as plt

def perceptron(inputs, weights, bias):
    """Calculates the output of a single perceptron utilizing a step activation function."""
    assert len(inputs) == len(weights), "Dimensionality mismatch between inputs and weights."
    
    # Calculate the weighted sum using the dot product
    weighted_sum = np.dot(inputs, weights) + bias

    # --- YOUR CODE HERE --- #
    # Apply the step activation function:
    # If weighted_sum is greater than or equal to 0, output 1
    # Otherwise, output 0
    # --- END YOUR CODE --- #

# Test functionality
test_inputs =[1, 0]
test_weights = [0.5, 0.5]
test_bias = -0.7
print(f"Test Output: {perceptron(test_inputs, test_weights, test_bias)}") 
```

<details>
<summary><b>Sample Solution & Code Explanation: Perceptron Function</b></summary>

**Solution:**
```python
def perceptron(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    if weighted_sum >= 0:
        return 1
    else:
        return 0
```

**Code Explanation for Beginners:**
*   `import numpy as np`: Imports the NumPy library, which handles numerical arrays and matrices efficiently.
*   `assert len(inputs) == len(weights)`: A safety check. It ensures we have exactly one weight for every input. If they don't match, the program throws an error.
*   `np.dot(inputs, weights)`: This performs the "dot product". Instead of writing a `for` loop to multiply `input[0] * weight[0] + input[1] * weight[1]...`, NumPy calculates this entire sum mathematically at the hardware level.
*   `if weighted_sum >= 0:`: This is the Heaviside step function. It acts as the activation threshold, converting the continuous numerical sum into a discrete binary output (1 or 0).
</details>

### 1.2 Modeling Logic Gates

By modifying the statistical weights and bias, the Perceptron can model linearly separable Boolean logic.

**Task:** Determine the appropriate weights and biases for the following logic gates.

**(a) OR Gate:** Outputs 1 if at least one input is 1.
```python
gate_inputs = [[0, 0], [0, 1],[1, 0], [1, 1]]

# --- YOUR TASK: Define weights and bias for OR --- #
weights_or = [None, None]
bias_or = None
# --- END YOUR TASK --- #

print("--- OR Gate Output ---")
for input_pair in gate_inputs:
    output = perceptron(input_pair, weights_or, bias_or)
    print(f"Input: {input_pair}, Output: {output}")
```

<details>
<summary><b>Sample Solution & Explanation: OR Gate</b></summary>

**Solution:**
```python
weights_or = [1.0, 1.0]
bias_or = -0.5
```
**Explanation:** If either input is `1`, the sum is at least `1.0`. Subtracting the bias of `0.5` yields `0.5`, which is $\ge 0$ (fires). If both are `0`, the result is `-0.5` (does not fire).
</details>

**(b) AND Gate:** Outputs 1 strictly when both inputs are 1.
```python
# --- YOUR TASK: Define weights and bias for AND --- #
weights_and = [None, None]
bias_and = None
# --- END YOUR TASK --- #

print("--- AND Gate Output ---")
for input_pair in gate_inputs:
    output = perceptron(input_pair, weights_and, bias_and)
    print(f"Input: {input_pair}, Output: {output}")
```

<details>
<summary><b>Sample Solution & Explanation: AND Gate</b></summary>

**Solution:**
```python
weights_and = [1.0, 1.0]
bias_and = -1.5
```
**Explanation:** By lowering the bias to `-1.5`, a single `1` input yields `1.0 - 1.5 = -0.5` (does not fire). Both inputs must be `1` to yield `2.0 - 1.5 = 0.5` (fires).
</details>


**(c) NOT Gate:** Outputs the inverse of a single input.
```python
not_inputs = [[0], [1]] 

# --- YOUR TASK: Define weight and bias for NOT --- #
weights_not =[None]
bias_not = None
# --- END YOUR TASK --- #

print("\n--- NOT Gate ---")
for input_val in not_inputs:
    # Make sure weights_not matches the input structure (single element)
    output = perceptron(input_val, weights_not, bias_not)
    print(f"Input: {input_val}, Output: {output}")
```
<details>
<summary><b>Sample Solution: NOT Gate</b></summary>

```python
weights_not = [-1.0]
bias_not = 0.5
```
</details>

### 1.3 Visualizing Linear Separability and the XOR Limitation

As established theoretically, a single Perceptron is restricted to defining linear decision boundaries. To understand why the single Perceptron fails, we can visualize the mathematical space. The code below plots the coordinates of the AND gate and the XOR gate on a 2D plane.

```python
# Define the input points as a NumPy array for easier slicing
inputs = np.array([[0, 0], [0, 1],[1, 0], [1, 1]])
x_coords = inputs[:, 0]
y_coords = inputs[:, 1]

# Define the expected outputs
outputs_and = np.array([0, 0, 0, 1])
outputs_xor = np.array([0, 1, 1, 0])

# Colors map: Index 0 is blue, Index 1 is red
colors = ['blue', 'red']

def plot_gate(gate_outputs, title):
    plt.figure(figsize=(5, 5)) 
    plt.title(title)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    
    # Plot each coordinate with the corresponding color based on its output
    for i in range(len(inputs)):
        plt.scatter(x_coords[i], y_coords[i], color=colors[gate_outputs[i]], s=150, zorder=3)
        
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.grid(True, zorder=0)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # Create legend manually
    import matplotlib.lines as mlines
    blue_marker = mlines.Line2D([],[], color='blue', marker='o', linestyle='None', markersize=10, label='Output 0')
    red_marker = mlines.Line2D([],[], color='red', marker='o', linestyle='None', markersize=10, label='Output 1')
    plt.legend(handles=[blue_marker, red_marker], loc='best')
    
    plt.show()

# Execute plotting
plot_gate(outputs_and, 'AND Gate Outputs (Linearly Separable)')
plot_gate(outputs_xor, 'XOR Gate Outputs (Non-Linearly Separable)')
```

<details>
<summary><b>Code Explanation: Matplotlib Plotting Logic</b></summary>

*   `inputs = np.array(...)`: Converts the nested Python lists into a multi-dimensional NumPy array.
*   `inputs[:, 0]`: This is NumPy slicing syntax. It means "take all rows (`:`), but only the element at index `0` (the first column)". This extracts all the X coordinates into a single list.
*   `plt.figure(figsize=(5, 5))`: Initializes a plotting canvas of 5x5 inches.
*   `color=colors[gate_outputs[i]]`: A clever trick. If the expected output is `0`, it accesses `colors[0]` (blue). If output is `1`, it accesses `colors[1]` (red).
*   `plt.scatter(...)`: Places a dot on the graph at the specified X and Y coordinate. `s=150` controls the size of the dot.
*   `plt.axhline` / `plt.axvline`: Draws the distinct black horizontal and vertical lines representing the X=0 and Y=0 axes.
</details>

> [!IMPORTANT]  
> **The XOR Challenge:** Observe the generated plots. You can draw a single straight line to separate the red dot from the blue dots in the AND graph. However, attempt to find weights and a bias for the perceptron function that satisfy XOR. You cannot. No single straight line can divide the diagonal red and blue dots in the XOR graph.


**Question:** Explain analytically why the Perceptron fails the XOR challenge based on the definition of its activation threshold ($\mathbf{w} \cdot \mathbf{x} + b \ge 0$).

<details>
<summary><b>Solution & Explanation</b></summary>
<br>
To solve XOR, the following system of inequalities must be true simultaneously:
<br>
1. $(0 \cdot w_1) + (0 \cdot w_2) + b < 0 \implies b < 0$<br>
2. $(1 \cdot w_1) + (0 \cdot w_2) + b \ge 0 \implies w_1 + b \ge 0$<br>
3. $(0 \cdot w_1) + (1 \cdot w_2) + b \ge 0 \implies w_2 + b \ge 0$<br>
4. $(1 \cdot w_1) + (1 \cdot w_2) + b < 0 \implies w_1 + w_2 + b < 0$
<br><br>
If we add inequalities 2 and 3, we get: $w_1 + w_2 + 2b \ge 0$. <br>
However, from inequality 1, we know $b < 0$. Therefore, $w_1 + w_2 + b$ must be strictly greater than $w_1 + w_2 + 2b$. <br>
This directly contradicts inequality 4 ($w_1 + w_2 + b < 0$). The system is mathematically unsolvable.

> [!IMPORTANT]  
> **The XOR Challenge:** Attempting to assign weights and a bias that satisfy the Exclusive-OR (XOR) logic table will result in a mathematical contradiction. No single linear hyperplane can divide the input coordinates `[0,1]` and `[1,0]` from `[0,0]` and `[1,1]`.

</details>

---

## Part 2: Multi-Layer Networks with Keras - Solving XOR

To process non-linearly separable data, network architecture must incorporate hidden layers with non-linear activation functions. This segment utilizes TensorFlow/Keras to build a Feedforward Neural Network (MLP).

### 2.1 Setup and Data Preparation

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

# XOR Dataset Formulation
X = np.array([[0, 0],[0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32) 
```

### 2.2 Building the MLP Architecture

The proposed architecture includes an input dimension of 2, one hidden layer utilizing ReLU (Rectified Linear Unit), and an output layer utilizing a Sigmoid function.

```python
# Initialize a sequential neural network model
model = keras.Sequential(name="MLP_XOR_Solver") 

# Add Hidden Layer
model.add(layers.Dense(units=4, activation='relu', input_shape=(2,)))

# Add Output Layer
model.add(layers.Dense(units=1, activation='sigmoid'))

# Print structural overview
model.summary()
```

<details>
<summary><b>Code Explanation: Keras Network Architecture</b></summary>

*   `keras.Sequential()`: Tells Keras we are building a model layer-by-layer, in a straight sequence from input to output.
*   `layers.Dense(...)`: Creates a "fully connected" layer, meaning every neuron in this layer connects to every neuron in the previous layer.
*   `units=4`: Specifies that this hidden layer will contain exactly 4 artificial neurons.
*   `activation='relu'`: Applies the Rectified Linear Unit function to the neurons. Without this, the network remains purely linear.
*   `input_shape=(2,)`: Required only for the very first layer. It tells the network to expect inputs with 2 features (Input 1 and Input 2).
*   `activation='sigmoid'`: Used in the final layer. It squashes the final network output into a probability score between 0 and 1, ideal for binary classification.
</details>

<details>
<summary><b>About input_shape in Keras 3</b></summary>

<br>
In **TensorFlow 2.16+ / Keras 3**, you should not pass `input_shape` into `Dense`. Instead, add an explicit `Input` layer first.

**Old (deprecated)**

```python
model = keras.Sequential()
model.add(layers.Dense(units=4, activation='relu', input_shape=(2,)))
```

**New (recommended with `model.add()`)**

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()

# Define input separately
model.add(keras.Input(shape=(2,)))

# Then add layers
model.add(layers.Dense(units=4, activation='relu'))
```

### Key idea

* `keras.Input(shape=(2,))` defines the input layer explicitly
* `Dense` layers now only define computation (not input shape)

</details>

**Question:** Review the output of `model.summary()`. How are the parameter (weight and bias) counts calculated for both the hidden and output layers?

<details>
<summary><b>Solution: Parameter Calculation</b></summary>
<br>
<b>Hidden Layer (4 neurons):</b>
<ul>
<li>Weights: 2 inputs $\times$ 4 neurons = 8 weights</li>
<li>Biases: 1 bias per neuron = 4 biases</li>
<li>Total: 12 parameters</li>
</ul>
<b>Output Layer (1 neuron):</b>
<ul>
<li>Weights: 4 inputs (from hidden layer) $\times$ 1 neuron = 4 weights</li>
<li>Biases: 1 bias per neuron = 1 bias</li>
<li>Total: 5 parameters</li>
</ul>
<b>Total Network Parameters:</b> 17
</details>

### 2.3 Compiling the Model

Before the network can learn, we must define *how* it learns.

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

<details>
<summary><b>Code Explanation: Keras Compilation</b></summary>

*   `model.compile()`: Prepares the mathematical backend for training.
*   `optimizer='adam'`: The algorithm used to adjust the weights. Adam is an advanced, highly efficient variation of Gradient Descent that automatically adjusts its own learning rate.
*   `loss='binary_crossentropy'`: The mathematical function used to calculate the error. Cross-entropy is the standard statistical measurement for comparing binary (0 or 1) predictions against actual targets.
*   `metrics=['accuracy']`: Tells the model to calculate and print the percentage of correct predictions during training so humans can monitor progress.
</details>

### 2.4 Training the Model

```python
# --- YOUR TASK: Define the number of epochs --- #
num_epochs = 1000 # Try 1000 to start
# --- END YOUR TASK --- #

history = model.fit(X, y, epochs=num_epochs, batch_size=4, verbose=0) 
print(f"Final Training Accuracy: {history.history['accuracy'][-1] * 100}%")
```

<details>
<summary><b>Code Explanation: Model Fitting</b></summary>

*   `model.fit(X, y, ...)`: The core command that initiates the Forward and Backward passes. It attempts to fit the network's weights to map inputs `X` to outputs `y`.
*   `epochs=1000`: An epoch is one complete pass through the entire dataset. Neural networks learn slowly via small adjustments, so processing the data 1,000 times is common.
*   `batch_size=4`: Determines how many data samples to process before the optimizer updates the weights. Since our XOR dataset only has 4 samples total, we process all of them in a single batch.
*   `verbose=0`: Suppresses the visual progress bar output to keep the console clean. Change to `1` to watch it train in real-time.
</details>

> [!NOTE]  
> If your model did not reach 100% accuracy, it likely became trapped in a "local minimum" due to unfortunate random initial weights. Re-run the model building and training cells to reset the weights and try again.

### 2.5 Evaluating the Model

```python
predictions = (model.predict(X) > 0.5).astype(int)
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted: {int(predictions[i])}, Expected: {int(y[i])}")
```


> [!WARNING]  
> **Why MLPs are theoretically suboptimal for XOR:** While an MLP *can* solve XOR, applying a continuous, gradient-based optimization algorithm (Adam/Gradient Descent) to a strictly discrete, Boolean problem is structurally inefficient. Because weights are initialized randomly, the network can easily converge into a "local minimum" (a mathematical valley) where it predicts 0.5 for all inputs, failing to learn the rule. Decision Trees or analytical Boolean logic are vastly superior for pure logic gate abstraction.

**Question:** Did the MLP achieve 100% accuracy on the XOR dataset during your trial? If it failed, what hyperparameter adjustments might facilitate convergence?

<details>
<summary><b>Solution & Explanation</b></summary>
<br>
If the model did not reach 100% accuracy, it likely became trapped in a local minimum due to poor initial weight randomization. 
<br><br>
<b>Corrective adjustments include:</b>
<br>1. Re-instantiating the model to force a new random weight initialization.
<br>2. Increasing the `num_epochs` to allow the optimizer more iterations to navigate the loss surface.
<br>3. Modifying the learning rate within the optimizer configuration.
</details>

---

## Part 3: Beyond XOR - Visualizing Complex Decision Boundaries

To thoroughly demonstrate why Multi-Layer Perceptrons require hidden layers and non-linear activations, we will evaluate the network on a circular coordinate dataset and visually plot the boundaries it creates.

**The Problem:** Given coordinates $(x, y)$, classify the point as `1` (red) if it falls within a circular radius, and `0` (blue) if it falls outside.

### 3.1 Generating and Training on Circular Data

```python
# Generate synthetic circular data
np.random.seed(42)
X_circle = np.random.uniform(-2, 2, (400, 2)) # 400 random coordinates between -2 and 2
# Target: 1 if inside radius 1.2, else 0 (Equation of a circle: x^2 + y^2 = r^2)
y_circle = (X_circle[:, 0]**2 + X_circle[:, 1]**2 <= 1.2**2).astype(int)

# 1. Define Model (Adding an extra hidden layer for geometric complexity)
circle_model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(2,)),
    layers.Dense(16, activation='relu'), 
    layers.Dense(1, activation='sigmoid')
])

# 2. Compile
circle_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Train
print("Training on circular dataset...")
circle_model.fit(X_circle, y_circle, epochs=300, batch_size=16, verbose=0)
print("Training Complete.")
```

### 3.2 Visualizing the Neural Network's Mind

How exactly does the MLP solve this? It bends the linear space. We can visualize this by asking the model to predict every possible coordinate on the graph, creating a colored "contour" map of its decisions.

```python
# Create a dense grid of coordinates across the entire graph space
xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 100), np.linspace(-2.5, 2.5, 100))

# Flatten the grid to feed into the model
grid_coordinates = np.c_[xx.ravel(), yy.ravel()]

# Predict the output for every single point on the grid
predictions = circle_model.predict(grid_coordinates, verbose=0)

# Reshape predictions back into the 2D grid shape
zz = predictions.reshape(xx.shape)

# Plotting the Decision Boundary
plt.figure(figsize=(7, 7))
plt.title("MLP Non-Linear Decision Boundary")

# plt.contourf colors the background based on the model's predictions
plt.contourf(xx, yy, zz, alpha=0.3, cmap='bwr')

# Plot the actual original dataset points on top
plt.scatter(X_circle[:, 0], X_circle[:, 1], c=y_circle, cmap='bwr', edgecolor='k')

plt.xlim(-2.5, 2.5); plt.ylim(-2.5, 2.5)
plt.xlabel("X Coordinate"); plt.ylabel("Y Coordinate")
plt.show()
```

<details>
<summary><b>Code Explanation: Meshgrids and Contour Plotting</b></summary>

*   `np.linspace(-2.5, 2.5, 100)`: Generates 100 evenly spaced numbers between -2.5 and 2.5.
*   `np.meshgrid(...)`: Takes the X and Y `linspace` arrays and creates a massive 2D grid matrix representing every single overlapping coordinate pair on the graph.
*   `xx.ravel()`: Flattens the 2D grid into a single 1D list so the neural network can process it. `np.c_` concatenates the X list and Y list side-by-side into pairs `[x, y]`.
*   `circle_model.predict(...)`: We pass thousands of synthetic coordinates into the trained model. It outputs a probability (0 to 1) for each coordinate.
*   `plt.contourf(...)`: A "filled contour" plot. It takes the grid (`xx`, `yy`) and the model's predictions (`zz`) and colors the background. The `cmap='bwr'` (Blue-White-Red) argument ensures values near 0 are blue, values near 1 are red, and the boundary (0.5) is white.
*   `c=y_circle`: Colors the actual data points based on their true labels, allowing us to see how well the model's colored background aligns with the actual dots.
</details>

> [!TIP]  
> **Observe the Output:** Look at the generated plot. The network has successfully drawn a nearly circular polygon around the center points. A single Perceptron could only draw a straight line through this space. The MLP, combining multiple neurons with ReLU activations, successfully segments non-linear data!

## Conclusion

Through this exercise, you have empirically verified the mathematical limitations of single-layer perceptrons utilizing visualization. By transitioning to a multi-layer framework utilizing Keras, you successfully optimized neural architectures capable of abstracting complex, spatial boundaries that are completely non-linearly separable.
# 2. Mathematical building blocks of neural networks

## 2.1 Concepts

**Layers:** Data processing module that you can think as a filter for data. They extract *representations* out of data fed to them.

**Dense layers:** Layers which are densely connected (also called *fully connected*)

**Softmax layer:** Layer that returns probability scores.

* **Loss function:** How the network will be able to measure its performance on the training data.
* **Optimizer:** Mechanism throughwhich the network will update itself based on the data and loss function.
* **Metrics to monitor during training and testing:** Generally it's accuracy.

**Overfitting:** The fact that machine-learning models tend to perform worse on new data than on their training data.

## 2.2 Data representations for neural networks

### Tensors

Container of data (almost always numerical). Generalization of matrices to an arbitrary number of dimentions.

* **Scalars (0D tensors):** Tensor that contains only one number.
* **Vectors (1D tensors):** An array of numbers. It has exactly one axis.
* **Matrices (2D tensors):** It has 2 axes. Visually interpreted as a rectangualr grid of numbers.
* **3D Vectrs and higher-dimentional tensors:** If you pack such matrices in a new array, you obtain a 3D tensor. Visually interpreted as a cube.

```python
>>> x = np.array([[[5, 78, 2, 34, 0],
[6, 79, 3, 35, 1],
[7, 80, 4, 36, 2]],
[[5, 78, 2, 34, 0],
[6, 79, 3, 35, 1],
[7, 80, 4, 36, 2]],
[[5, 78, 2, 34, 0],
[6, 79, 3, 35, 1],
[7, 80, 4, 36, 2]]])
>>> x.ndim
3
```

**Dimensionality:** Can denote either the number of entries along a specific axis or the number of axes in a tensor.

#### Key atributes

A tensor is defined by 3 key attributes:

* **Number of axes (rank):** . This is also called the tensor’s `ndim` in Python libraries such as Numpy.

  ```python
  >>> print(train_images.ndim)
  3
  ```
* **Shape:** Tuple of integers that describes how many dimensions the tensor has along each axis.

  ```python
  >>> print(train_images.shape)
  (60000, 28, 28)
  ```
* **Data type:** Type of data contained in the tensor; for instance, a tensor’s type could be `float32`, `uint8`, `float64`, and so on. Usually called `dtype` in Python libraries.

  ```python
  >>> print(train_images.dtype)
  uint8
  ```

#### Manipulating tensors in Numpy

**Tensor slicing:** Selecting a range of data in a tensor.

```python
my_slice = train_images[10:100]
```

In general, you may select between any two indices along each tensor axis.

```python
my_slice = train_images[:, 14:, 14:]
```

#### Data batches

Deep-learning models don’t process an entire dataset at once; rather,
they break the data into small batches.

When considering a batch tensor, the first axis (axis 0) is called the **batch axis** or **batch dimension**.

## 2.3 Tensor operations

Layers can be interpreted as a function like this:

```python
output = relu(dot(W, input) + b)
```

### **Element-wise operations**

Operations that are applied independently to each entry in the tensors being considered. Highly optimized.

```python
import numpy as np
z = x + y #Element-wise addition
z = np.maximum(z, 0.) #Element-wise relu
```

**Broadcasting:** When possible, and if there’s no ambiguity, the smaller tensor will be broadcasted to match the shape of the larger tensor. Broadcasting consists of two steps:

1. Axes (called broadcast axes) are added to the smaller tensor to match the ndim of the larger tensor.
2. The smaller tensor is repeated alongside these new axes to match the full shape of the larger tensor

### Tensor dot

Most useful. It combines entries in the input tensors.

```python
import numpy as np
z=np.dot(x,y)
```

To understand dot-product shape compatibility, it helps to visualize the input and output tensors by aligning them.


![1678416608774](image\Notes\1678416608774.png)

### Tensor reshaping

Reshaping a tensor means rearranging its rows and columns to match a target shape.
Naturally, the reshaped tensor has the same total number of coefficients as the initial tensor.

```python
x = x.reshape((x,y,...))
```

**Transposition:** Exchanging rows and columns.

## 2.4 Gradient-based optimization

Each neural layer transforms input data as follows:

```python
output = relu(dot(W, input) + b)
```

**W** and **b** are tensors that are attributes of the layer. They are **weigths** or **trainable parameters** of the layer. They contain the information learned by the network from exposure to training data.

**Random initialization:** Fil the initial weights with small random numbers.

### Training loop

1. Draw a batch of training samples *x* and corresponding targets `y`.
2. Run the network on `x` to obtain predictions `y_pred`.
3. Compute the loss of the networks on the batch. Measure the mismatch between `y_pred` and `y`.
4. Update all the weights of the network in a way that slightly reduces the loss on this batch.

### Gradient

Is the derivative of a tensor operation.

If the data inputs `x` and `y`, this can be interpreted as a function mapping values of `W` to loss values:

```python
loss_value = f(W)
```

The derivative of `f` in the point `W` is a tensor `gradient(f)(W)`, where each coefficient `gradient(f)(W)[i,j]` indicate the direction and magnitude of the change in `loss_value` when modifying `W[i,j]`.

`gradient(f)(W)` can be interpreted as the tensor describing the *curvature* of `f(W)` around the current `W`.

You can reduce the value of `f(W)` by moving `W` in the opposite direction from the gradient. For example, `W1 = W0 - step * gradient(f)(W0)` (where `step` is a small scaling factor).

![1679175493660](image\Notes\1679175493660.png)

### Stochastic gradient descent

If you update the weigths in the opposite direction from the gradient, the loss will be a little less every time. The step 4 can be efficiently implemented as:

4. Compute the gradient of the loss with regard to the network's parameters (A *backward pass*)
5. Move the parameters a little in the opposite direction from the gradient.

This is called *mini-batch stochastic gradient descent.*

**Stochastic->** Each batch of data is drawn at *random*.

The **step** can't be too *small* or it will take too many iterations or bestuck in local minimum. It can't be too *large* or the updates may be too random.

#### Momentum

A useful mental image here is to think of the optimization process as a small ball rolling down the loss curve. If it has enough momentum, the ball won’t get stuck in a ravine and will end up at the global minimum.

**Momentum** is implemented by moving the ball at each step based not only on the current slope value (current acceleration) but also on the current velocity (resulting from past acceleration). In practice, this means updating the parameter w based not only on the current gradient value but also on the previous parameter update, such as in this naive implementation:

```python
past_velocity = 0
momentum = 0.1 # Constant momentum factor
while loss > 0.01: # Optimization loop
	w, loss, gradient = get_current_parameters()
	velocity = past_velocity * momentum + learning_rate * gradient
	w = w + momentum * velocity - learning_rate * gradient
	past_velocity = velocity
	update_parameter(w)
```

### Backpropagation

Starts with the final loss value and works backward from thetop layers to the botton layers, applying the *chain rule* to compute the contribution that each parameter had in the loss value.


# 3. Getting started with neural networks

## 3.1 Anatomy of a neural network

Training a neural network revolves around:

* **Layers**, which are combined into a **network**.
* The **input data** and corresponding **targets**.
* The **loss function**, which defines the feedback signal used for learning.
* The **optimizer**, which dtermines how learning proceeds

![1679189190342](image\Notes\1679189190342.png)

### Layers

A **layer** is a data-processing module that takes as input one or more tensors and outputs one or more tensors.

Different layers are appropiate for different tensor formarts and different types of data processing

* Simple vector data, 2D shape -> **densely connected layers**
* Sequence data, 3D shape -> **recurrent layers**
* Image data, 4D shape -> **2D convolution layers**

Building deep-learning models is done by clipping together compatible layers to form userful data-transformation pipelines.

**Layer compatibility** refers specifically to the fact that every layer will only accept input tensors of a certain shape and will return output tensors of a certain shape.

### Models

A **model** is a directed, acyclic graph of layers.

There are many topologies:

* Linear stack of layers (most common)
* Two-branch networks
* Multihead networks
* Inception blocks

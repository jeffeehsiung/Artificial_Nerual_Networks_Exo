# Artificial Neural Networks and Deep Learning

## 1. Exploring Neural Networks for Regression and Classification

### Objective

In this lab session, we delve into the mechanisms of neural networks for regression and classification tasks, emphasizing training methods, generalization, and optimization algorithms which concepts are presented in section 1.3 "In the Wide Jungle of the Training Algorithms" and 1.4 "A Personal Regression Exercise."

### Introduction

This report investigates various gradient-based optimization algorithms and their impact on the training and generalization of neural models. Additionally, we present a hands-on experiment in approximating a nonlinear function using a neural network tailored to a unique dataset, highlighting the intricacies of model selection and evaluation.

### Methodology

The first part of our study scrutinizes different optimization algorithms using PyTorch implementations. We examine the role of noise in optimization processes, compare vanilla gradient descent with its stochastic and accelerated variants, assess the influence of network size on optimizer selection, and distinguish between epochs and time for algorithm speed evaluation.

In the second part, we tackle a regression problem, approximating an unknown nonlinear function based on a given dataset of 13,600 datapoints. Our approach involves constructing a personal dataset from five nonlinear functions, designing a neural network architecture, and evaluating its performance on a separate test set.

## Section 1.3: A small model for a small dataset

- **Answer with plots and tabular numerical data**

### Q1. What is the impact of the noise parameter in the example with respect to the optimization process?

Idea: define a list of noise and find online if there is a way to evaluate if the noise is too large compared to input. Is there a way to quantify the relationship between noise to signal ratio, and the impact on the optimizer?

The impact of noise can be quantified by listing a selection of noise, calculating for each selection the signal-to-noise ratio, fitting the model given the signal integrated with noise, and plotting the training-validation loss curve and learning curve.

- **SNR table**

   | Noise | SNR [dB]        | Residual    |
   |-------|-----------------|-------------|
   | 0.1   | tensor(9.7257)  | tensor(0.0106) |
   | 0.3   | tensor(0.5533)  | tensor(0.0879) |
   | 0.9   | tensor(-8.9133) | tensor(0.7776) |
   | 1.0   | tensor(-10.1901)| tensor(1.0433) |
   | 1.3   | tensor(-12.3810)| tensor(1.7278) |
   | 1.6   | tensor(-14.2371)| tensor(2.6492) |
   | 1.9   | tensor(-15.4338)| tensor(3.4897) |
   | 2.0   | tensor(-15.9377)| tensor(3.9190) |

**Impact of Noise on Optimization:**
- **Increased Difficulty:** Higher noise levels in the data can make the optimization process more challenging. Noise introduces variability in the loss landscape, making it harder for the optimizer to find a clear path toward the minimum.
- **Risk of Overfitting:** With more noise, there's a greater risk that the model may overfit to the noisy data, capturing the noise as if it were a meaningful pattern. This reduces the model's ability to generalize to new, unseen data.

**Using LBFGS with Noisy Data:**
- LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is an optimization algorithm designed for solving smooth and convex optimization problems and is particularly well-suited for quasi-Newton methods.
- **Sensitivity to Noise:** LBFGS, being a second-order optimization method, is more sensitive to the quality of the gradient information. Noise can affect the Hessian approximation (which LBFGS uses to guide its updates), potentially leading to less stable updates.
- **Adaptations:** Implementing mechanisms such as line search strategies (like 'strong_wolfe') helps LBFGS adapt its step size in response to the noise, attempting to ensure that each step improves the loss in a meaningful way, despite the noise.

**Hypothetical Outcomes and Visualizations:**
- **Training Loss Plot:** 

- **Squared Residuals:** The plot of squared residuals (the squared differences between predictions and actual values) would likely show higher values on average, indicating greater prediction error due to the noise.

### Experimental Result

1. **Training and Validation Loss**
   - Description: Track the training and validation loss over epochs. An increasing gap between training and validation loss might indicate overfitting, which can be exacerbated by noise.
   - Quantification: Calculate the difference or ratio between training and validation loss. A larger difference suggests that noise may be negatively impacting generalization.
   - **Results**:

   |     | Training error | Residual | Lowest loss | Best epoch | SNR [dB]  |
   |-----|----------------|----------|-------------|------------|-----------|
   | 0.1 | 0.006351       | 0.006351 | 0.006352    | 1999       | 12.424053 |
   | 0.3 | 0.054652       | 0.054652 | 0.054655    | 1999       | 5.529366  |
   | 0.9 | 0.480521       | 0.480521 | 0.480558    | 1999       | 2.698386  |
   | 1.0 | 0.615826       | 0.615826 | 0.615904    | 1999       | 2.443620  |
   | 1.3 | 1.058633       | 1.058633 | 1.058636    | 1999       | 2.283705  |
   | 1.6 | 1.632100       | 1.632100 | 1.632197    | 1999       | 2.244913  |
   | 1.9 | 2.265662       | 2.265662 | 2.265848    | 1999       | 2.367871  |
   | 2.0 | 2.524409       | 2.524409 | 2.524705    | 1999       | 1.954540  |
   
   ```python
   - We would expect to see potentially more fluctuations in the training loss over epochs with higher noise levels. The convergence might also be slower or less smooth compared to training on less noisy data.
   ```

2. **Predicted Surface vs. True Surface:** 
   ```python
   The predicted surface plot would likely show more deviation from the true underlying function due to the noise. This deviation would manifest as a less smooth surface or one that doesn't capture the true pattern as cleanly.
   ```
3. **Squared Residuals:** 
   The plot of squared residuals (the squared differences between predictions and actual values) would likely show higher values on average, indicating greater prediction error due to the noise.
   ```python
   squared residuals plot to be added
   ```
4. **Model Accuracy**
   - Description: Evaluate the model's accuracy (or other relevant metrics) on both the training set and an unseen test set.
   ```python
   F1 score to be added
   ```

### Q2. How does (vanilla) gradient descent compare with respect to its stochastic and accelerated versions?
#### Vanilla Gradient Descent
In the context of neural network training, **Vanilla Gradient Descent** refers to the simplest form of gradient descent optimization algorithm. It is a first-order iterative optimization algorithm for finding the minimum of a function. Here's a basic explanation:

- **Objective**: The goal of gradient descent is to minimize a loss function, which measures the difference between the predicted output of the neural network and the actual target values. The loss function landscape can be thought of as a surface with hills and valleys, where each point on this surface represents a particular set of model parameters (weights and biases), and the elevation represents the loss value for those parameters.

- **How It Works**: Vanilla gradient descent updates all model parameters simultaneously, taking steps proportional to the negative of the gradient (or approximate gradient) of the loss function with respect to those parameters. This is akin to descending down the surface of the loss function to find its minimum value, which corresponds to the most optimal model parameters.

- **Update Rule**: 
    The update rule for the parameters in gradient descent is given by:

    `θ = θ - η ∇_θJ(θ)`

    where:
    - `θ` represents the parameters of the model,
    - `η` is the learning rate (a small, positive hyperparameter that determines the size of the steps),
    - `∇_θJ(θ)` is the gradient of the loss function `J(θ)` with respect to the parameters.

- **Characteristics**:
    - The term "vanilla" indicates that this is the most basic form of gradient descent, without any modifications or optimizations like momentum or adaptive learning rates.
    - It involves a full computation of the gradient using the entire dataset, which makes it computationally expensive and slow for large datasets.
    - It can be slow to converge, especially in loss function landscapes that are shallow or have many plateaus, saddle points, or local minima.

#### Stochastic Gradient Descent (SGD)
In contrast, **Stochastic Gradient Descent (SGD)** and **Accelerated versions** (such as Momentum, Nesterov Accelerated Gradient, Adam, etc.) introduce various optimizations to improve the convergence speed, efficiency, or stability of the training process. For example, SGD updates the model parameters using the gradient computed from a randomly selected subset of the data (a mini-batch) rather than the entire dataset, significantly speeding up the computation and allowing for more frequent updates.

### Q3. How does the size of the network impact the choice of the optimizer?
The size of the neural network can indeed impact the choice of the optimizer. 

1. **Memory Usage**: Some optimizers require more memory because they need to store additional parameters or states. For example, Adam, RMSProp, and other adaptive learning rate methods store an exponentially decaying average of past gradients. This can be problematic for very large networks or for devices with limited memory.

2. **Convergence Speed**: For large networks, the speed of convergence becomes critical. Stochastic Gradient Descent (SGD) with momentum or adaptive learning rate methods like Adam can converge faster than vanilla SGD, which can be beneficial for large networks.

3. **Generalization**: Some research suggests that simpler optimizers like SGD may generalize better for larger networks, while adaptive methods might lead to overfitting. However, this can be problem-dependent and is still an active area of research.

4. **Computational Overhead**: Optimizers like Adam, RMSProp have additional computational overhead compared to SGD due to the calculation of moving averages of gradients or squared gradients. For large networks, this overhead can be significant.
### Experimental Result
In the experiment, we can compare the convergence speed and generalization performance of different optimizers (e.g., SGD, Adam, RMSProp) on networks of varying sizes. We can also measure the memory usage and computational time for each optimizer and network size combination to understand the practical implications of the choice. 
To evaluate the impact of network size on the choice of optimizer based on convergence speed and generalization performance, the code is modified to record the convergence speed exemplified by the best epoch number given different network sizes and optimizers.

   | Optimizer Number | Optimizer Type | Learning Rate | Momentum | Nesterov | Max Iter | Line Search Fn |
   |------------------|----------------|---------------|----------|----------|----------|----------------|
   | 1                | SGD            | 0.05          | -        | -        | -        | -              |
   | 2                | SGD            | 0.1           | -        | -        | -        | -              |
   | 3                | SGD            | 0.1           | 0.9      | True     | -        | -              |
   | 4                | Adam           | -             | -        | -        | -        | -              |
   | 5                | LBFGS          | 1             | -        | -        | 1        | strong_wolfe   |


   | Network Size | Optimizer 1 | Optimizer 2 | Optimizer 3 | Optimizer 4 | Optimizer 5 |
   |--------------|-------------|-------------|-------------|-------------|-------------|
   | 10           | 2445        | 2208        | 1985        | 2500        | 2499        |
   | 50           | 2475        | 2400        | 2465        | 2499        | 2500        |
   | 100          | 2359        | 2460        | 2430        | 2499        | 2500        |

```python
# Example code to record the best epoch number for different network sizes and optimizers
Explain the results
```

### Q4. Discuss the difference between epochs and time to assess the speed of the algorithms. What can it mean to converge fast ?
When evaluating the performance of optimization algorithms in training neural networks, both the number of epochs and the time taken are crucial metrics, but they measure different aspects of the learning process. Understanding the distinction between them is essential for accurately assessing algorithm speed and efficiency.

#### Epochs

- **Definition**: An epoch is a single pass through the entire training dataset. Completing one epoch means the algorithm has used every sample in the dataset once to update the model's parameters.
- **Convergence Speed**: In the context of epochs, convergence speed refers to the number of epochs required for an algorithm to reach a certain level of accuracy or to minimize the loss function to a predefined threshold. Fewer epochs needed for convergence generally indicate a faster learning algorithm, assuming all other factors are equal.
- **Evaluation Metric**: Using epochs as a metric allows us to assess the efficiency of the learning process in terms of dataset utilization. It provides insights into how quickly the model learns from the entire dataset.

#### Time

- **Definition**: Time refers to the actual duration taken by the algorithm to reach convergence, measured in units of time such as seconds or minutes.
- **Convergence Speed**: When considering time, convergence speed is about how quickly an algorithm can achieve a specified level of performance. This measure takes into account not only the efficiency of learning from the data (as measured by epochs) but also the computational complexity of the algorithm, including the time it takes to process each epoch.
- **Evaluation Metric**: Time as a metric provides a practical understanding of an algorithm's performance, especially in real-world applications where computational resources and time are limited. It includes the effects of implementation details, hardware efficiency, and algorithmic complexity.

#### Comparing Epochs and Time for Convergence

- **Epochs for Learning Efficiency**: Evaluating algorithms based on the number of epochs emphasizes the model's ability to learn from data. It abstracts away from computational aspects, focusing on the learning algorithm's theoretical efficiency. However, it doesn't account for the time each epoch takes, which can vary significantly between different algorithms or even different implementations of the same algorithm.
- **Time for Practical Efficiency**: Considering the time to converge provides a holistic view of an algorithm's efficiency, incorporating both the learning efficiency and the computational cost. It's particularly relevant in applied settings where time and computational resources are constraints.

#### Fast Convergence

- **In Terms of Epochs**: Fast convergence in terms of epochs means the algorithm requires fewer passes through the dataset to reach its convergence criteria. This can be indicative of a more efficient learning process but doesn't account for the computational cost per epoch.
- **In Terms of Time**: Fast convergence in terms of time means the algorithm reaches its convergence criteria more quickly in real time. This measure is influenced by both the number of epochs required and the computational efficiency of the algorithm.

#### Conclusion

Fast convergence can mean different things depending on whether we're discussing epochs or time. In practice, the best measure depends on the specific context and constraints of the application. For research or situations where the learning process itself is under scrutiny, epochs might be the more relevant metric. In contrast, for practical applications where computational resources and time are limited, the actual time to convergence is often more critical.

### Q5: A Bigger Model: Number of Parameters in the Model

The model's total number of parameters can be calculated by examining each layer's contribution. The formula to calculate the number of parameters for Conv2D and Dense layers is given by:

- **Conv2D Layers**: $(\text{kernel width} \times \text{kernel height} \times \text{input channels} + 1) \times \text{number of filters}$
- **Dense Layers**: $(\text{input units} + 1) \times \text{output units}$

Let's calculate the number of parameters for the provided model:

1. **First Conv2D Layer**: $(3 \times 3 \times 1 + 1) \times 32 = 320$
2. **Second Conv2D Layer**: $(3 \times 3 \times 32 + 1) \times 64 = 18,496$
3. **Dense Layer**: After flattening and max pooling, the calculation requires knowing the output size from the previous layer. Assuming the max pooling layers do not overlap, the dimension after two max pooling layers (each with a stride of 2) on a 28x28 image would be reduced to $7 \times 7$ for each of the 64 filters from the second Conv2D layer. Hence, $(7 \times 7 \times 64 + 1) \times 10 = 31,750$

The dropout layer does not add any parameters; it only acts during training by randomly setting a fraction of the input units to 0 at each update during training time to prevent overfitting.

To calculate the exact number of parameters, you could use the `model.summary()` method in Keras, which prints a summary representation of your model, including the number of parameters (trainable and non-trainable) at each layer and the total.

   <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
   ┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">    Param # </span>┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
   │ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">26</span>, <span style="color: #00af00; text-decoration-color: #00af00">26</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │        <span style="color: #00af00; text-decoration-color: #00af00">320</span> │
   ├─────────────────────────────────┼───────────────────────────┼────────────┤
   │ max_pooling2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
   ├─────────────────────────────────┼───────────────────────────┼────────────┤
   │ conv2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">11</span>, <span style="color: #00af00; text-decoration-color: #00af00">11</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │     <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
   ├─────────────────────────────────┼───────────────────────────┼────────────┤
   │ max_pooling2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
   ├─────────────────────────────────┼───────────────────────────┼────────────┤
   │ flatten_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1600</span>)              │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
   ├─────────────────────────────────┼───────────────────────────┼────────────┤
   │ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1600</span>)              │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
   ├─────────────────────────────────┼───────────────────────────┼────────────┤
   │ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)                │     <span style="color: #00af00; text-decoration-color: #00af00">16,010</span> │
   └─────────────────────────────────┴───────────────────────────┴────────────┘
   </pre>

   <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">34,826</span> (136.04 KB)
   </pre>

   <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">34,826</span> (136.04 KB)
   </pre>

   <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
   </pre>


### Q6: A Bigger Model: Replacing the Optimizer
- **Adam to SGD**: Adam is an adaptive learning rate optimizer that combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems. SGD (Stochastic Gradient Descent), on the other hand, maintains a single learning rate for all weight updates and the learning rate does not change during training. Changing Adam to SGD might slow down the convergence, and you might need to fine-tune the learning rate and possibly add momentum to achieve similar performance. Despite these differences, excellent performance can still be achieved with SGD, especially with careful tuning of its hyperparameters.

- **SGD to Adadelta**: Adadelta is an extension of AdaGrad aiming to reduce its aggressive, monotonically decreasing learning rate. It does so by restricting the window of accumulated past gradients to a fixed size. Adadelta does not require a default learning rate setting, making it easier to configure since it adapts over time. The particularity of Adadelta is that it seeks to reduce the learning rate's diminishing learning rates effect, making it more robust for various problems. However, its performance compared to Adam or SGD can vary depending on the task and specific model architecture.

Regarding the particularities of each optimizer and their impact on performance:

- **SGD** may require more careful tuning of the learning rate and may benefit from momentum to accelerate convergence in relevant directions. It's simpler but can be very effective with the right settings.

- **Adadelta** is designed to reduce the learning rate's aggressive decay and does not require a learning rate to be specified, making it easier to configure. It's adaptive and aims to address the diminishing learning rates problem of AdaGrad.

#### Experimental Result
- SGD can perform very well on a wide range of problems, especially with momentum. 
- Adadelta, being adaptive, might perform better in scenarios where the optimal learning rate changes over time.

   | Optimizer  | Test Loss | Test Accuracy |
   |------------|-----------|---------------|
   | ADAM       | 0.025648  | 0.9909        |
   | SGD        | 0.022971  | 0.9917        |
   | Adadelta   | 0.022916  | 0.9918        |

## Section 1.4: A Personal Regression Exercise
### Q1: Why spliting Training and Testing Dataset. Plot the surface associated to the training set.

The splitting should be done randomly to ensure that both datasets are representative of the overall data distribution.

1. **Explain the Point of Having Different Datasets for Training and Testing**: This is important to evaluate the generalization capability of the model. The training dataset is used to fit the model, i.e., to allow the model to learn the underlying patterns in the data. The testing dataset, which consists of unseen data, is used to assess how well the model performs on data it hasn't encountered before. If you only used a single dataset for both training and testing, you wouldn't be able to tell if the model had simply memorized the data (overfitting) or if it had learned to generalize from the patterns it had seen during training.

2. **Plot the Surface Associated to the Training Set**: Once the `T_new` target is constructed using your student number as weights for the different functions, you can plot the surface associated with the training set. This plot will visualize the new function that your neural network will try to approximate.

   <p align="center">
   <img src="image/training_set_triSurface.png" width="300" height="200">
   <br>
   <em>Figure: Surface associated to the Training Set.</em>
   </p>

### Q2: The Neural Network Architecture Hyperparameters Tuning
Build and train your feedforward neural network. To that end, you must perform an adequate model selection on the training set. Investigate carefully the architecture of your model: number of layers, number of neurons, learning algorithm and transfer function. How do you validate your model ?

systematically explore a range of different model architectures, learning rates, and other hyperparameters to find the best performing model on your validation set. 

1. **Model Selection**: The model selection process involves experimenting with different hyperparameters to find the combination that yields the best performance on the training set. This includes:
   - **Number of Layers**: Experiment with different numbers of hidden layers and units in each layer to find the architecture that best captures the underlying patterns in the data.
   - **Learning Algorithm**: Try different optimization algorithms (e.g., SGD, Adam, RMSProp) and learning rates to find the one that converges most effectively.
   - **Transfer Function**: Experiment with different activation functions (e.g., ReLU, Sigmoid, Tanh) to find the one that best captures the non-linear relationships in the data.

In practice, you may want to use more sophisticated techniques and libraries designed for hyperparameter optimization, like `keras-tuner` or `scikit-learn`'s `GridSearchCV` for larger spaces of parameters.

2. **Validation**: To validate the model, you can use a separate validation set (distinct from the training and testing sets) to assess the model's performance during training. This can help you monitor for overfitting and ensure that the model generalizes well to unseen data. Additionally, you can use techniques like cross-validation to assess the model's performance across different subsets of the training data.

#### Experimental Result
A set of hyperparameters was created for model selection, using the validation set as a basis. The model with the lowest validation loss was selected as the best model.

   | Hyperparameter      | Options          |
   |---------------------|------------------|
   | Number of layers    | 1, 2, 3          |
   | Units per layer     | 3, 10, 20        |
   | Activation function | tanh, relu       |
   | Learning rate       | 0.01, 0.001, 0.0001 |

**The best model is found as follows:**

   | Best Model Parameters | Value                  |
   |-----------------------|------------------------|
   | Layers                | 3                      |
   | Units                 | 20                     |
   | Activation            | tanh                   |
   | Learning Rate         | 0.01                   |
   | Validation Loss       | 0.002038267906755209   |
   | Training Error        | 0.0024579891469329596  |
   | Validation Error      | 0.002038267906755209   |

   <p align="center">
   <img src="image/training_validation_loss_curve.png" width="300" height="200">
   <br>
   <em>Figure: Training and validation loss curve.</em>
   </p>


### Q3: The Neural Network Training and Testing
Evaluate the performance of your selected network on the test set. Plot the surface of the test set and the approximation given by the network. Explain why you cannot train further. Give the final MSE on the test set.

1. **Performance Evaluation**: 
Prediction on test set gives the Final Mean Squared Error (MSE) on the test set as a quantitative measure of the model's performance.

   | Metric | Value |
   |--------|-------|
   | MSE    | 0.23111077279740277 |

2. **Surface Plot and Approximation**: 
   The test set surface and the approximation given by the network is visualized as follows:
      <p align="center">
      <img src="image/test_set_triSurface.png" width="300" height="200">
      <br>
      <em>Figure: Surface of the test set and the approximation given by the network.</em>
      </p>
   The delta between prediction and actual value is visualized as follows:
      <p align="center">
      <img src="image/delta_surface.png" width="300" height="200">
      <br>
      <em>Figure: Delta between prediction and actual value.</em>
      </p>

3. **Training Limitations**: 
- The best model has 2 hidden layers, 20 units in each layer, relu activation, and 0.01 learning rate.
- The model has a low training, along with the lowest validation error. Additionally, from the test result, it shows that the test error is also low.
- The model has a good generalization performance and cannot train further. If training further, despite lowered training error, the validation error would likely increase, indicating overfitting.


### Q4: The Neural Network Regularization Strategies
Describe the regularization strategy that you used to avoid overfitting. What other strategy can you think of ?

1. **Regularization Strategy**: Regularization techniques like L1 and L2 regularization, dropout, and early stopping can be used to avoid overfitting. Describe the specific regularization strategy you used and how it helped improve the model's generalization performance.

2. **Other Strategies**: Discuss other strategies that can be used to avoid overfitting, such as data augmentation, batch normalization, and model ensembling.

# Artificial Neural Networks and Deep Learning

## 2. Recurrent Nerual Networks

## Section 2.1: Hopfield Network

### Q1. Hopfield network with target patterns [1, 1], [−1, −1] and [1, −1] and the corresponding number of neurons. Simulate finding attractors after a sufficient number of iterations for both Random and High-symmetry input vectors. Evaluate the obtained attractors.

1. **Attractor values**
- **Random Inputs: Attractors vs. Targets**
   <p align="center">
   <img src="image/random_inputs.png" width="300" height="200">
   <br>
   <em>Figure: Time evolution in the state space of random input.</em>
   </p>

    - The final states of the random input vectors converge to one of the target patterns (`[1, 1]`, `[-1, -1]`, and `[1, -1]`). 
    - In some cases, the network converges to `[-1, 1]`, which is not one of the original target patterns. This state is known as a spurious state or a false attractor, which is a byproduct of the network's dynamics. It is a result of the symmetric nature of the weights in the Hopfield network, which inherently supports both the target pattern and its inverse as stable states.

    <p align="center">
    <img src="image/energy_convolution_random.png" width="300" height="200">
    <br>
    <em>Figure: Energy evolution of the state of random input.</em>
    </p>

- **Symmetric Inputs: Attractors vs. Targets**

   <p align="center">
   <img src="image/symmetric_inputs.png" width="300" height="200">
   <br>
   <em>Figure: Time evolution in the state space of symmetric input.</em>
   </p>
    - The final states for symmetric inputs e.g. `[1, 0]`, `[0, 1]`, `[-1, 0]`, and `[0, -1]` are not among the orignal target patterns but ranges between them. The attractors represent states of high symmetry and are sometimes referred to as points of unstable equilibrium in the Hopfield network. 
    - The network does not converge to a specific attractor but rather to a state that is equidistant from multiple attractors. This is especially noticeable in the last case, where the final state `[0, 0]` is exactly in the middle of all the attractors and is essentially an unstable point.
- **Convergence**: 
    - The network typically converges to one of the target patterns or their inverses. In the case of symmetric inputs, the convergence is to states of high symmetry, which are close to multiple attractors.

    <p align="center">
    <img src="image/energy_convolution_symmetric.png" width="300" height="200">
    <br>
    <em>Figure: Energy evolution of the state of symmetric input.</em>
    </p>

2. **Unwanted attractors**
- The presence of unwanted attractors such as `[-1, 1]` arises from the interactions between the neurons in the network. The network's energy function has local minima at these points, which are not explicitly trained for but are emergent properties of the network.

3. **Number of iterations to reach the attractor**
- The number of iterations to reach an attractor varies depending on the initial state and the network's energy landscape. For some initial states, convergence may be quick, while for others, especially those starting near the decision boundary between basins of attraction, it may take longer.
    | Input Type        | Average Iterations to Reach Attractor |
    |-------------------|---------------------------------------|
    | Random Inputs     | 10.0                                  |
    | Symmetric Inputs  | 0.0                                   |
- **Random Inputs**: 
    - The number of iterations required for the network to converge to an attractor varies between 1 and 10. The convergence is typically fast, and the network reaches a stable state within a few iterations.
- **Symmetric Inputs**: 
    - The number of iterations required for the network to converge to an attractor is also low, typically between 1 and 10. The convergence is fast, and the network reaches a stable state within a few iterations.

4. **Stability of the attractors**
- The attractors `[1, 1]`, `[-1, -1]`, and `[1, -1]` are stable as they are the intended patterns that the network was trained on. The spurious attractor `[-1, 1]` is also stable, although it is not a desired state.  For symmetric inputs, the network does not converge to the target patterns but rather to intermediate states that are not stable attractors.

### Q2. Hopfield network with target patterns [1, 1, 1], [−1, −1, −1], [1, −1, 1] and the corresponding number of neurons. Simulate finding attractors after a sufficient number of iterations for both Random and High-symmetry input vectors. Evaluate the obtained attractors.

1. **Attractor values**
- **Random Inputs: Attractors vs. Targets**
   <p align="center">
   <img src="image/random_inputs_3D.png" width="300" height="200">
   <br>
   <em>Figure: Time evolution in the state space of 3D random input.</em>
   </p>

    - **Final State vs. Target Patterns**: The network converges to states that are either directly one of the target patterns or closely related. This indicates the network's ability to recall the stored patterns from various initial states.
    - **Stability and Recall Accuracy**: The stable final states ([1, -1, -1], [-1, -1, 1], and [1, 1, 1]) match the target patterns, demonstrating the network's associative memory properties. It suggests that these patterns are stable attractors in the network's energy landscape.
    - **Variability in Convergence**: Different initial states lead to convergence to different target patterns, showcasing the network's sensitivity to initial conditions and its capacity to differentiate between distinct attractors.

    <p align="center">
    <img src="image/energy_convolution_random_3D.png" width="300" height="200">
    <br>
    <em>Figure: Energy evolution of the state of 3D random input.</em>
    </p>

- **Symmetric Inputs: Attractors vs. Targets**
   <p align="center">
   <img src="image/symmetric_inputs_3D.png" width="300" height="200">
   <br>
   <em>Figure: Time evolution in the state space of 3D symmetric input.</em>
   </p>

    - **Near-symmetric States**: The final states for symmetric inputs often don't match exactly with the target patterns. Instead, they are near states of high symmetry, indicating that these inputs are near the boundary regions between the basins of attraction of the target patterns. This phenomenon illustrates the concept of "energy landscape" in Hopfield networks, where certain initial states can lead the network to converge to intermediate states close to multiple attractors.
    - **Stability of Symmetric States**: The appearance of states like [1, 0.06045472, -0.06045472] suggests that these symmetric or near-symmetric inputs do not strongly converge to one specific target pattern but rather to a state influenced by the surrounding attractors. This is particularly notable in a high-dimensional space, where the energy landscape can be complex.

    <p align="center">
    <img src="image/energy_convolution_symmetric_3D.png" width="300" height="200">
    <br>
    <em>Figure: Energy evolution of the state of 3D symmetric input.</em>
    </p>


2. **Average Number of Iterations to Reach an Attractor**
    - **Rapid Convergence for Random Inputs**: An average of 10 iterations to reach an attractor for random inputs suggests that the network can quickly stabilize to a memorized pattern from a variety of starting points.
    - **Immediate Convergence for Symmetric Inputs**: The reported average of 0 iterations for symmetric inputs might be misleading or an artifact of how convergence was measured. It likely indicates that these inputs are already very close to or within the basin of attraction of their final states from the beginning.

3. **Overall Interpretation**
    - **Existence of Spurious States**: The appearance of states not directly matching the targets, especially in symmetric simulations, points to the existence of spurious states or mixed states due to the complex interplay of attractors in the network's configuration space.

    In practical terms, these results illustrate the Hopfield network's capabilities and limitations as a content-addressable memory system, its sensitivity to initial states, and the influence of the network's structure on its dynamic behavior.

### Q3.Create a higher dimensional Hopfield network which has as attractors the handwritten digits from 0 to 9. Test the ability of the network to correctly retrieve these patterns when some noisy digits are given as input to the network. Try to answer the below questions by playing with these two parameters:
- noise level represents the level of noise that will corrupt the digits and is a positive number. 
- num iter is the number of iterations the Hopfield network (having as input the noisy digits) will run.
1. **Is the Hopfield model always able to reconstruct the noisy digits? If not why? What is the influence of the noise on the number of iterations?**
    <!-- title of the table: Iteration 50 Noise Level vs. Accuracy -->
    #### Iteration 50 Noise Level vs. Accuracy
    | Noise Level | Accuracy (%) |
    |-------------|--------------|
    | 0.0         | 100.0        |
    | 0.1         | 100.0        |
    | 0.2         | 100.0        |
    | 0.3         | 100.0        |
    | 0.4         | 100.0        |
    | ...         | ...          |
    | 9.6         | 50.0         |
    | 9.7         | 30.0         |
    | 9.8         | 80.0         |
    | 9.9         | 30.0         |
    | 10.0        | 40.0         |

    <p align="center">
    <img src="image/digits_hopfield.png" width="300" height="200">
    <br>
    <em>Figure: Hopfield digit reconstruction with noise on Iteration set to 50.</em>
    </p>

    - **Hopfield Model's Reconstruction Ability**:
        - The Hopfield model is not always able to reconstruct the noisy digits perfectly due to the presence of spurious states and the network's energy landscape. The influence of noise on the number of iterations is significant, as higher noise levels can lead to longer convergence times or even prevent the network from reaching the correct attractor.
        - **Effect of Noise on Reconstruction**: 
            - Low noise levels may allow the network to converge to the correct attractor with minimal distortion. However, as the noise level increases, the network may struggle to recover the original pattern due to the interference caused by the noisy inputs.
            - The network may converge to a spurious state or a mixed state that is a combination of the original pattern and the noise, especially when the noise level is high.
        - **Influence of Noise on Convergence Time**:
            - Higher noise levels can lead to longer convergence times as the network needs more iterations to overcome the noise and reach the correct attractor.
            - The network may get stuck in local minima or oscillate between states when the noise level is high, resulting in prolonged convergence times or failure to converge.
        - **Trade-off between Noise and Convergence**:
            - There is a trade-off between the noise level and the convergence time. Higher noise levels can make it more challenging for the network to reconstruct the original pattern, leading to longer convergence times or convergence to incorrect states.
            - The network's ability to reconstruct the noisy digits depends on the noise level, the network's architecture, and the specific attractors present in the network.

## Section 2.2: Time-Series Prediction with Santa Fe Laser Dataset
### Q1. Implement a time-series prediction MLP network to predict the Santa Fe Laser dataset. Explore the effect of lag, validation size, validation folds, and the number of neurons in the hidden layer on the prediction performance. Discuss the results obtained.

1. **Effect of Lag on Prediction Performance**
    - **Lag Parameter**: The lag parameter determines the number of previous time steps used as input features for the prediction model. A higher lag value captures more historical information but may lead to increased complexity and longer training times.
    - **Impact of Lag on Performance**:
        - **Low Lag Values**: A low lag value may not capture sufficient historical context, leading to poor prediction accuracy and generalization.
        - **Optimal Lag Values**: There exists an optimal lag value that balances the trade-off between capturing relevant information and model complexity. This value can be determined through experimentation and validation.
        - **High Lag Values**: Excessive lag values can introduce noise and irrelevant information, leading to overfitting and reduced prediction performance.
2. **Effect of Validation Size and Folds on Prediction Performance**
    When setting the validation size and the number of validation folds in the context of the `prepare_timeseries` function you've described, you're essentially configuring how to split your time series data into training and validation sets for the purpose of cross-validation. These parameters directly influence the robustness of your model evaluation and the reliability of performance metrics. Let's understand their impact and how to choose them:

    - **Validation Size**:
        - The `validation_size` parameter specifies the size of the validation set. This is often defined as a fraction of the total dataset size. For instance, a `validation_size` of 0.2 means that 20% of the data at the end of the time series will be used for validation, while the remaining 80% will be available for training.
        - Choosing a validation size is a balance between having enough data to train your model effectively and having enough data to validate the model's performance reliably. A common split ratio for time series data is 80/20 or 70/30 (training/validation).

    - **Validation Folds**:
        - The `validation_folds` parameter specifies the number of folds or splits to use in the validation process. 
        - The number of folds is observed to be the number of `available data points` divided by the the `sum` of the `validation size` and the `gap`. The gap is the number of time steps between the training and validation sets.
        - This is relevant for cross-validation, where the model is trained and validated several times, each time with a different fold acting as the validation set and the remaining data used for training.
        - More folds mean more training/validation cycles, which can provide a more accurate estimate of model performance but at the cost of increased computational time and complexity. For time series, the number of folds is often limited by the chronological nature of the data; you cannot randomly shuffle time series data as you might with cross-sectional data without disrupting temporal dependencies.
        - A typical choice might be 5 or 10 folds for cross-validation, but the best choice depends on the size of your dataset and the computational resources at your disposal.

    - **Relationship Between Parameters and Data Size**:
        - The total size of your input data constrains the choices for `validation_size` and `validation_folds`. Specifically, you need enough data points to create meaningful splits. For instance, if you have a very small dataset, using a large number of folds or a very small validation size might not be practical because each individual fold might not contain enough data to be representative or to train the model effectively.
        - It's also essential to consider the "gap" parameter used in time series splitting, corresponding to the lag in your data. The gap ensures that there's a temporal buffer between your training and validation sets, which can affect how you set your validation size and folds.

    - **Recommendations**:
        - Start with standard practices, such as a validation size of 20-30% and 5-10 folds, and adjust based on the performance and specific characteristics of your time series data (e.g., seasonality, length).
        - Ensure your validation set is large enough to capture the key patterns in the data and that each fold in cross-validation has a meaningful amount of data.
        - Remember, the objective is to validate the model's ability to generalize to unseen data effectively, so the settings should reflect the balance between a robust training process and an accurate estimation of performance on new data.

3. **Effect of Number of Neurons in Hidden Layer on Prediction Performance**
    Model performance with different lags and numbers of neurons (hidden units in the MLP) was investigated. Based on a set of lags (e.g., 1, 5, 10, 20) and the number of neurons (e.g., 5, 10, 25, 50), a grid search was performed to evaluate the model's performance, or  mean squared error (MSE), using cross-validation. The results were analyzed to determine the optimal combination of lag and neuron count that yielded the lowest MSE on the test set. The result records are discussed and visualized to understand the influence of each parameter on model performance.

    - **Influence of Neurons on Model Complexity**:
        - **Low Neuron Count**: A low number of neurons may limit the model's capacity to capture complex patterns in the data, potentially leading to underfitting.
        - **Optimal Neuron Count**: There exists an optimal number of neurons that balances model complexity and performance. This value can be determined through experimentation and validation.
        - **High Neuron Count**: Too many neurons can lead to overfitting, where the model learns noise in the training data rather than the underlying patterns, resulting in poor generalization to unseen data.

    - **Interplay Between Lag and Neurons**:
        - The choice of lag and the number of neurons in the hidden layer can interact to influence the model's performance. A larger lag might require more neurons to capture the additional historical context, but this could also introduce noise or irrelevant information. It's essential to balance these aspects effectively to achieve the best predictive performance.

    - **Model Evaluation and Selection**:
        - The grid search approach allows for a systematic evaluation of different combinations of lag and neuron count to identify the best-performing model. By analyzing the results, you can gain insights into how these parameters affect the model's ability to predict the Santa Fe Laser dataset accurately.
        - Visualizing the learning curves, MSE values, and other relevant metrics can provide a comprehensive view of the model's performance and help in selecting the optimal hyperparameters for the MLP network.
        - **Learning Curve without Early Stopping**: 
            <p align="center">
            <img src="image/MLP_training_fold_4_lag_5_H_10.png" width="400" height="150">
            <br>
            <em>Figure: Loss curve: 4 validation folds, 5 lags, 10 hidden units.</em>
            </p>
            <p align="center">
            <img src="image/MLP_training_fold_4_lag_5_H_25.png" width="400" height="150">
            <br>
            <em>Figure: Loss curve: 4 validation folds, 5 lags, 25 hidden units.</em>
            </p>
            <p align="center">
            <img src="image/MLP_training_fold_4_lag_5_H_50.png" width="400" height="150">
            <br>
            <em>Figure: Loss curve: 4 validation folds, 5 lags, 25 hidden units.</em>
            </p>
            <p align="center">
            <img src="image/MLP_training_fold_4_lag_10_H_5.png" width="400" height="150">
            <br>
            <em>Figure: Loss curve: 4 validation folds, 10 lags, 5 hidden units.</em>
            </p>
            <p align="center">
            <img src="image/MLP_training_fold_4_lag_10_H_10.png" width="400" height="150">
            <br>
            <em>Figure: Loss curve: 4 validation folds, 10 lags, 10 hidden units.</em>
            </p>
            <p align="center">
            <img src="image/MLP_training_fold_4_lag_10_H_25.png" width="400" height="150">
            <br>
            <em>Figure: Loss curve: 4 validation folds, 10 lags, 25 hidden units.</em>
            </p>
            <p align="center">
            <img src="image/MLP_training_fold_4_lag_10_H_50.png" width="400" height="150">
            <br>
            <em>Figure: Loss curve: 4 validation folds, 10 lags, 50 hidden units.</em>
            </p>
            <p align="center">
            <img src="image/MLP_training_fold_4_lag_15_H_25.png" width="400" height="150">
            <br>
            <em>Figure: Loss curve: 4 validation folds, 15 lags, 25 hidden units.</em>
            </p>
            <p align="center">
            <img src="image/MLP_training_fold_4_lag_15_H_50.png" width="400" height="150">
            <br>
            <em>Figure: Loss curve: 4 validation folds, 15 lags, 50 hidden units.</em>
            </p>

        - **Learning Curve MSE without Early Stopping**: 
            | Lag | H=5   | H=10  | H=25  | H=50  |
            |-----|-------|-------|-------|-------|
            | 1   | 0.628 | 0.644 | 0.650 | 0.622 |
            | 5   | 0.112 | 0.046 | 0.058 | 0.087 |
            | 10  | 0.180 | 0.068 | 0.176 | 0.205 |
            | 15  | 0.136 | 0.149 | 0.275 | 0.532 |

            - The best performance without early stoping on the validation set is achieved with lag being 5 and H being 10: MSE = 0.046
            - Yet, if with early stoping, the best performance may be achieved with lag 5 and H 25 or 50, or with lag being 10 and H being 25, or at last with lag being 15 and H being 25.
            - Typically, more neurons can capture more complex patterns, but also run the risk of overfitting. Similarly, a larger lag might provide more historical context for predictions, but could also introduce noise or irrelevant information. The best combination will balance these aspects effectively.
            - Consider the trade-offs between model complexity, prediction accuracy, and generalization when selecting the optimal hyperparameters for the MLP network. The goal is to find a model that can effectively predict the Santa Fe Laser dataset while avoiding overfitting or underfitting.

    - **Discussion and Conclusion**:
    #### **Exercise 1**
    Given the learning curve shape, combinations of lag and hidden unit where the training and validation loss converges were selected to predict on test sets. If there is early stopping, despite the lowest MSE without early stopping indicating the combination of lag 5 and H 10 being the best parameter set as lags with higher number in combination with H being 25 or 50 tends to overfit at the end of the epochs, combinatoins of lag being 10 or 15 with H being 25 would perform better than combination of lag being 5 and H being 10 if with early stopping. By evaluating these combinatoin via prediction,  Which combination of parameters gives the best performance (MSE) on the test set?

    - **Lag 5 H 25: The MSE on the test set is: 3908.419**
        <p align="center">
        <img src="image/predictoin_lag5_h25.png" width="400" height="150">
        <br>
        <em>Figure: MLP Lag=5 H=25 prediction results on continuation of Santa Fe laser datase .</em>
        </p>
    - **Lag 5 H 50: The MSE on the test set is: 2997.659**
        <p align="center">
        <img src="image/predictoin_lag5_h50.png" width="400" height="150">
        <br>
        <em>Figure: MLP Lag=5 H=50 prediction results on continuation of Santa Fe laser datase .</em>
        </p>
    - **Lag 10 H 25: The MSE on the test set is: 3065.473**
        <p align="center">
        <img src="image/predictoin_lag10_h25.png" width="400" height="150">
        <br>
        <em>Figure: MLP Lag=10 H=25 prediction results on continuation of Santa Fe laser datase .</em>
        </p>
    - **Lag 15 H 25: The MSE on the test set is: 2940.037**
        <p align="center">
        <img src="image/predictoin_lag15_h25.png" width="400" height="150">
        <br>
        <em>Figure: MLP Lag=15 H=25 prediction results on continuation of Santa Fe laser datase .</em>
        </p>
    - **Lag 15 H 50: The MSE on the test set is: 6488.547**
        <p align="center">
        <img src="image/predictoin_lag15_h50.png" width="400" height="150">
        <br>
        <em>Figure: MLP Lag=15 H=50 prediction results on continuation of Santa Fe laser datase .</em>
        </p>

### Q2. Do the same for the LSTM model and explain the design process. What is the effect of changing the lag value for the LSTM network?
The LSTM model was implemented to predict the Santa Fe Laser dataset, and the design process involved configuring the model architecture, hyperparameters, and training settings. The impact of changing the lag value on the LSTM network was explored to understand how historical context influences prediction performance.

- Performance on the validation set is achieved with:
    - Lag being 15 and H being 25 performs better on the cross-validatoin set among the lag set of 1, 5, 10, 15 and number of hiddent units set of 5, 10, 25, 50, with MSE on cross-valdiation being the lowest, 0.110.
    - Lag being 15 and H being 25, the MSE on the test set is: 3410.111
        <p align="center">
        <img src="image/lstm_prediction.png" width="400" height="150">
        <br>
        <em>Figure: LSTM Lag=15 H=25 prediction results on continuation of Santa Fe laser datase .</em>
        </p>

### Q3. Compare the results of the recurrent MLP with the LSTM. Which model do you prefer and why?
Both the LSTM and MLP models have been used to predict the continuation of a time series from the Santa Fe laser dataset with the same lag of 15 and number of hidden units at 25.
        <p align="center">
        <img src="image/lstm_vs_mlp.png" width="400" height="150">
        <br>
        <em>Figure: Lag=15 H=25 LSTM vs. MLP prediction results on continuation of Santa Fe laser datase .</em>
        </p>
In comparing the two:

1. **LSTM Results**: The LSTM model appears to track the test data more closely. The peaks and troughs of the predicted values (orange line) follow the test data (blue line) with a higher degree of accuracy, particularly in capturing the rhythm and magnitude of the test data's oscillations.

2. **MLP Results**: The MLP model also captures the overall trend but seems to struggle with the finer details of the test data's fluctuations. It fails to predict the peaks and troughs accurately after about timestep 1060, deviating significantly from the test data.

**Preference**:
Based on the visual comparison, the LSTM model would be preferred for the following reasons:

- **Temporal Dependencies**: LSTMs are specifically designed to handle sequences with long-term dependencies. The LSTM seems to be leveraging its recurrent connections to better remember and predict the sequence of the laser intensity.

- **Prediction Accuracy**: The LSTM's predictions are closer to the test data, suggesting that it is better at generalizing from the training data to predict unseen sequences.

- **Stability**: After timestep 1060, the MLP predictions become erratic and unstable, whereas the LSTM maintains a consistent prediction pattern throughout the entire sequence.

In time series prediction tasks, especially with complex patterns and potential long-term dependencies, LSTMs often outperform traditional MLPs due to their recurrent structure. However, it's also important to consider factors such as training time, computational resources, and the specific characteristics of the dataset when choosing between the two models for practical applications.

# Artificial Neural Networks and Deep Learning

## 3. Deep Feature Learning 

## Section 3.1: Autoencoders and Stacked Autoencoders

### Q1. Conduct image reconstruction on synthetic handwritten digits dataset (MNIST) using an autoencoder. Note that you can tune the number of neurons in the hidden layer (encoding dim) of the autoencoder and the number of training epochs (n epochs) so as to obtain good reconstruction results. Can you improve the performance of the given model?.

Using a separate validation set for parameter tuning and reserving the test set for final evaluation ensures an unbiased performance measure on unseen data. In the process, training data is split into a new training set (80%) and a validation set (20%), with the latter used for hyperparameter tuning. The model undergoes training on this adjusted training set, validation against the validation set, and is ultimately evaluated on the test set to assess its generalization capability.

Based on the validation set performance given the following hyperparameters configuration:
| Batch Size | Encoding Dimension | Epochs | Average Validation Loss |
|------------|--------------------|--------|-------------------------|
| 16         | 32                 | 40     | 0.017948554020375013    |
| 16         | 32                 | 60     | 0.012706649725635847    |
| 16         | 64                 | 40     | 0.004797367513800661    |
| 16         | 64                 | 60     | 0.005407857708943387    |
| 16         | 128                | 40     | 0.001803275318040202    |
| 16         | 128                | 60     | 0.001841558615056177    |
| 32         | 32                 | 40     | 0.01410618870705366     |
| 32         | 32                 | 60     | 0.013888560553391775    |
| 32         | 64                 | 40     | 0.004698568860068917    |
| 32         | 64                 | 60     | 0.005495807031790415    |
| 32         | 128                | 40     | 0.0017972165880103905   |
| 32         | 128                | 60     | 0.0017599450452253221   |
| 64         | 32                 | 40     | 0.015392365475495657    |
| 64         | 32                 | 60     | 0.014030455440282821    |
| 64         | 64                 | 40     | 0.005479766167700291    |
| 64         | 64                 | 60     | 0.005293305454154809    |
| 64         | 128                | 40     | 0.0017778251422569155   |
| 64         | 128                | 60     | 0.0017278431582575043   |


The best hyperparameters configuration for the autoencoder model is as follows:
- Batch Size: 64
- Encoding Dimension: 128
- Epochs: 60

The average validation loss for this configuration is 0.0017278431582575043, indicating a good reconstruction performance on the validation set. This configuration was selected based on the lowest validation loss achieved during hyperparameter tuning.

loss of the test set evaluation of the model with the best hyperparameters configuration is 0.000227, compared to the oriiginal test loss of 0.001470 given the initial hyperparameters configuration of batch size 32, encoding dimension 32, and number of epochs 20. This indicates an improvement in the model's performance on unseen data after hyperparameter tuning.

### Q2. Conduct image classification on MNIST using an stacked autoencoder. Are you able to obtain a better result by changing the size of the network architecture? What are the results before and after fine-tuning? What is the benefit of pretraining the network layer by layer?

The baseline stacked autoencoder model with the following hyperparameters configuration:
- Batch Size: 128
- number of epoches layer wise: 10
- number of epochs classifier: 10
- number of epochs fine-tuning: 10
has accuracy 0.926 on the entire test set after fine-tuning phase.

If we chnage the batch size to 64, the number of epoches layer wise to 60, the number of epochs classifier to 60, and the number of epochs fine-tuning to 60, the accuracy of the model on the entire test set after fine-tuning phase is 0.949. 

Additionally, if the last layer of hidden dimension of the autoencoder is decrease from 256 to 128, the accuracy of the model on the entire test set after fine-tuning phase is 0.954.

The results before and after fine-tuning show that changing the size of the network architecture, increasing the number of epochs, and decreasing the hidden dimension of the autoencoder can improve the performance of the stacked autoencoder model on the MNIST dataset. The accuracy of the model on the entire test set increased from 0.926 to 0.954 by adjusting these hyperparameters, which hyperparameters configuration is identical to the best combindation for single autoencoder model.

## Section 3.2: Convolutional Neural Networks
### Q1. Answer the following questions: Consider the following 2D input matrix.
    
    ```
    X = [
        [2, 5, 4, 1],
        [3, 1, 2, 0],
        [4, 5, 7, 1],
        [1, 2, 3, 4]
        ]
    ```
#### Q1.1. Calculate the output of a convolution with the following 2x2 kernel with no padding and a stride of 2.
    
        ```
        K = [
            [1, 0],
            [0, 1]
            ]
        ```
- The output matrix is a 1x1 matrix with the value 1. To answer Q1.1, let's perform the convolution operation using the given input matrix $X$ and kernel $K$, with no padding and a stride of 2. Convolution involves sliding the kernel over the input matrix, computing the element-wise product of the kernel and the part of the input it covers at each step, and summing up these products to produce a single output value for each position the kernel can fit. The stride determines how many positions we move the kernel each time, and no padding means we don't add any borders to the input matrix.

Given $X$ and $K$:

$X = \begin{bmatrix} 2 & 5 & 4 & 1 \\ 3 & 1 & 2 & 0 \\ 4 & 5 & 7 & 1 \\ 1 & 2 & 3 & 4 \end{bmatrix}$

$K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$

With a stride of 2 and no padding, we will calculate the convolutional operation for each applicable position of $K$ over $X$.

For Q1.2, the dimensionality of the output of a convolutional layer can be determined by the formula:

$O = \frac{W - K + 2P}{S} + 1$

where:
- $O$ is the output size (height/width),
- $W$ is the input size (height/width),
- $K$ is the kernel size (height/width),
- $P$ is the padding on each side (total padding divided by 2 if it's uniform),
- $S$ is the stride.

This formula calculates the size of one dimension (height or width), and you would use the respective sizes for $W$, $K$, and $P$ for height and width to calculate each dimension of the output separately.

Let's perform the convolution operation for Q1.1 and then discuss the dimensionality further with the formula in mind.

For Q1.1, the output of the convolution operation with the given 2x2 kernel, no padding, and a stride of 2 on the input matrix is:

$
\begin{bmatrix}
3 & 4 \\
6 & 11
\end{bmatrix}
$

#### Q1.2. How do you in general determine the dimensionality of the output of a convolutional layer?
Regarding Q1.2, the general formula to determine the dimensionality of the output of a convolutional layer is given by:

$
O = \frac{W - K + 2P}{S} + 1
$

where $O$ is the output size for one dimension (height or width), $W$ is the input size for the same dimension, $K$ is the kernel size (assuming square kernels for simplicity), $P$ is the padding applied on each side of the input in that dimension, and $S$ is the stride of the convolution.

$O = \frac{4 - 2 + 2\cdot 0}{2} + 1 = 2$

Therefore, the output matrix is a 2x2 matrix with the values:

$
\begin{bmatrix}
3 & 4 \\
6 & 11
\end{bmatrix}
$

In the specific case of the convolution we just calculated, with no padding ($P=0$) and a stride of 2, the formula simplifies to just considering the input size, kernel size, and stride. Padding wasn't a factor here, but it plays a crucial role in many convolutional neural network designs to control the output size and preserve spatial dimensions through layers.

#### Q1.3. What benefits do CNNs have over regular fully connected networks?

CNNs have several benefits over regular fully connected networks, especially when dealing with data that has spatial or temporal structure, such as images, audio, and text. Some of the key advantages of CNNs include:

1. **Local Connectivity**: CNNs leverage local connectivity by using convolutional layers that apply filters to small regions of the input data. This allows the network to capture spatial hierarchies and patterns efficiently, reducing the number of parameters compared to fully connected networks.

2. **Parameter Sharing**: CNNs share weights across different regions of the input, enabling the network to learn spatially invariant features. This sharing of parameters helps generalize the learned features and reduces overfitting, especially in tasks with limited training data.

3. **Translation Invariance**: CNNs are inherently translation-invariant due to the use of convolutional layers. This property allows the network to recognize patterns regardless of their position in the input, making CNNs robust to translations and distortions in the data.

4. **Hierarchical Feature Learning**: CNNs learn hierarchical representations of features, starting from low-level features (e.g., edges, textures) in early layers to high-level features (e.g., objects, shapes) in deeper layers. This hierarchical feature learning enables CNNs to capture complex patterns and relationships in the data.

5. **Spatial Preservation**: CNNs preserve the spatial structure of the input data through convolutional and pooling layers. This spatial preservation is crucial for tasks where the spatial arrangement of features is important, such as image recognition and segmentation.

6. **Efficient Training**: CNNs are computationally efficient due to weight sharing and local connectivity, making them suitable for large-scale datasets and complex tasks. Additionally, techniques like transfer learning and data augmentation can further improve training efficiency and generalization.

Overall, CNNs are well-suited for tasks involving spatial or temporal data, where capturing local patterns, spatial hierarchies, and translation invariance is essential. Their architectural design and properties make them powerful tools for image processing, computer vision, natural language processing, and other domains where structured data is prevalent.

### Q2. The file cnn.ipynb runs a small CNN on the handwritten digits dataset (MNIST). Use this script to investigate some CNN architectures. Try out different amounts of layers, combinations of different kinds of layers, number of filters and kernel sizes. Note that emphasis is not on experimenting with batch size or epochs, but on parameters specific to CNNs. Pay close attention when adjusting the parameters for a convolutional layer as the dimensions of the input and output between layers must align. Discuss your results. Please remember that some architectures will take a long time to train.

### Initial CNN Architecture Test Accuracy:
The initial CNN architecture of four layers, each input channels respectively being 1, 16, 32, and 32, as well as output channels being 16, 32, 32, 32, and kernel size starts from 3 and to 2 at the end, with padding ebing 1, provided in the script achieves a test accuracy of approximately 0.98 as indicated in below figure.  This serves as a baseline for evaluating the performance of modified architectures.
<p align="center">
<img src="image/cnn_baseline_test_acc.png" width="300" height="200">
<br>
<em>Figure: Initial CNN Architecture Baseline Test Accuracy</em>
</p>

To explore different Convolutional Neural Network (CNN) architectures on the MNIST dataset using the provided script, the number of layers, types of layers (convolutional layers, pooling layers, fully connected layers), the number of filters, and kernel sizes are experimented with. These adjustments can help us understand how each parameter affects the network's ability to learn from the data. Here are some key points and suggestions for experimenting with CNN architectures:

### 1. Adjusting the Number of Convolutional Layers:
- **Experiment**: Increase or decrease the number of convolutional layers.
- **Expectation**: Adding more layers might allow the network to learn more complex features, but too many layers can lead to overfitting or increased computational cost.

### 2. Modifying the Number of Filters:
- **Experiment**: Change the number of filters in the convolutional layers.
- **Expectation**: More filters can capture more features, but similar to layers, an excessive number might cause overfitting or unnecessary computational expense.

### 3. Changing Kernel Sizes:
- **Experiment**: Use different kernel sizes in the convolutional layers.
- **Expectation**: Larger kernels might capture more global features, whereas smaller kernels may capture more local features. The choice of kernel size can affect the granularity of the features learned.

### 4. Incorporating Pooling Layers:
- **Experiment**: Add pooling layers (e.g., MaxPooling) after some convolutional layers.
- **Expectation**: Pooling layers can reduce the spatial size of the representation, making the network more efficient and reducing the chance of overfitting.

### 5. Adjusting the Fully Connected Layers:
- **Experiment**: Change the number and size of fully connected layers.
- **Expectation**: This can impact the network’s capacity to combine features into higher-order representations.

### 6. Experimenting with Activation Functions:
- **Experiment**: Try different activation functions (e.g., ReLU, LeakyReLU, Tanh).
- **Expectation**: The choice of activation function can affect the convergence rate and the ability of the network to model complex functions.

### 7. Applying Batch Normalization:
- **Experiment**: Add or remove batch normalization layers.
- **Expectation**: Batch normalization can improve training stability and speed up convergence.

In the experiment, the filter numbers of 16, 32, 64, 128, 256, kernel size of 3, and 5, appending one layer of leaky ReLU activation function, and one layer of batch normalization are tested. With numbers of trainning epoch, batch size, as well as shuffle condition remains the same as baseline, the test accuracy of the model with the best hyperparameters configuration is 0.991.

### Interpretation of Results:
- The initial CNN architecture achieved a test accuracy of approximately 0.98 on the MNIST dataset.
- By experimenting with different architectures, including varying the number of filters, kernel sizes, and activation functions, the test accuracy was improved to 0.991.
- The modifications, such as increasing the number of filters and incorporating batch normalization, deepen the network, and the use of pooling, have led to a slight improvement in the model's performance as indiated in below figure.

<p align="center">
<img src="image/cnn_exp_test_acc.png" width="300" height="200">
<br>
<em>Figure: Initial CNN Architecture Baseline Test Accuracy</em>
</p>


## Section 3.3: Self-Attention and Transformers
### Q1. Please run both the NumPy and PyTorch implementations of the self-attention mechanism. Can you explain briefly how the dimensions between the queries, keys and values, attention scores and attention outputs are related? What do the query, key and value vectors represent? Note that the attention mechanism will also be discussed in lecture 11.

To understand how the dimensions between queries, keys, values, attention scores, and attention outputs are related, let's dive into the core of the self-attention mechanism, both in the NumPy and PyTorch implementations provided.

### The Self-Attention Mechanism:

1. **Queries (Q), Keys (K), and Values (V)**:
   - These are derived from the input data (X) by projecting it through different weight matrices (Wq for Q, Wk for K, and Wv for V).
   - **Dimensions**: If the input X has dimensions `[n, d]` where `n` is the number of tokens (or samples) and `d` is the dimensionality of each token, the dimensions of Q, K, and V after projection are `[n, d_k]`, `[n, d_k]`, and `[n, d_v]` respectively, where `d_k` and `d_v` are dimensions decided by the projection matrices `Wq`, `Wk`, and `Wv`.

2. **Attention Scores**:
   - The attention scores are computed by taking a dot product of the query matrix with the key matrix transpose, followed by a scaling factor (usually `1/sqrt(d_k)`). This results in a matrix of dimensions `[n, n]`, representing the attention score between each pair of tokens in the input.
   - **Dimensions**: `[n, n]`.

3. **Attention Output**:
   - The final attention outputs are computed by multiplying the attention scores with the value matrix. This operation effectively weighs the value vectors by how well the corresponding keys and queries match.
   - **Dimensions**: The output dimension is `[n, d_v]`, the same as the value matrix.

### Representation of Q, K, V:
- **Query (Q)** vectors are projections of the input data that are used to score how each key matches with every other key, indicating the importance of the inputs.
- **Key (K)** vectors are used together with the query vectors to compute the attention scores. The scores determine how much focus should be put on other parts of the input data for each token.
- **Value (V)** vectors are also projections of the input data that are aggregated according to the computed attention scores to produce the final output.

### Importance in Attention Mechanism:
The attention mechanism allows the model to focus on different parts of the input sequence when producing each token in the output sequence. This is crucial for tasks such as translation, where the relevance of input tokens can vary depending on the context. By learning how to weigh input tokens differently, the model can capture dependencies and relationships in the data more effectively.

The process of applying self-attention in the context of transformers involves multiple such attention heads, allowing the model to jointly attend to information from different representation subspaces at different positions. This parallel attention processing capability significantly enhances the model's ability to understand and generate sequences, leading to the powerful performance of transformer-based models in a wide range of applications.

### Q2. Please train the Transformer on the MNIST dataset. You can try to change the architecture by tuning dim, depth, heads, mlp dim for better results. You can try to increase or decrease the network size and see whether it will influence the prediction results much. Note that ViT can easily overfit on small datasets due to its large capacity. Discuss your results under different architecture sizes.

In the provided script, the Transformer model is trained on the MNIST dataset, and the architecture is experimented with by tuning the following hyperparameters:
- `dim`: The dimensionality of the model (default: 64).
- `depth`: The number of transformer blocks (default: 6).
- `heads`: The number of attention heads in the multi-head attention mechanism (default: 8).
- `mlp_dim`: The dimensionality of the feedforward network inside the transformer blocks (default: 128).

The average test accuracy of the model with the best hyperparameters configuration is 0.9834 and the average test loss is 0.0746 given 20 training epochs. This serves as a baseline for evaluating the performance of modified architectures.

The goal is to explore how changing the architecture size affects the model's performance on the MNIST dataset. By adjusting the `dim`, `depth`, `heads`, and `mlp_dim` parameters of the `ViT` class, impact on the model's performance by systematical adjustment of these parameters can be outlined and evaluated. Multiple Configurations including the base configuration, increased dim and mlp_dim, and decreased depth and heads are tested to observe the effect of these changes on the model's performance.

```python
configurations = [
    {'dim': 64, 'depth': 6, 'heads': 8, 'mlp_dim': 128},  # Base configuration
    {'dim': 128, 'depth': 6, 'heads': 8, 'mlp_dim': 256},  # Increased dim and mlp_dim
    {'dim': 32, 'depth': 6, 'heads': 8, 'mlp_dim': 64},  # Decreased dim and mlp_dim
    {'dim': 64, 'depth': 8, 'heads': 16, 'mlp_dim': 128},  # Increased depth and heads
    {'dim': 64, 'depth': 3, 'heads': 4, 'mlp_dim': 128},  # Decreased depth and heads
    {'dim': 128, 'depth': 8, 'heads': 8, 'mlp_dim': 256},  # Increased dim, depth, and mlp_dim
    {'dim': 128, 'depth': 8, 'heads': 16, 'mlp_dim': 256},  # Increased dim, depth, heads, and mlp_dim
]
```
### Discussion Points

After running the experiments for different configurations, you'll want to compare and discuss several points:

- **Model Performance**: How does each configuration affect the accuracy on the MNIST test set? Is there a noticeable difference between smaller and larger models?
- **Overfitting**: Larger models with more parameters might overfit the relatively simple MNIST dataset. How does changing the model size affect overfitting, and what signs of overfitting are observable (if any)?
- **Training Dynamics**: Observe the loss curves during training. Do larger models converge faster? Is there any instability in training for specific configurations?
- **Computational Cost**: Larger models will generally take longer to train. Is the increase in computational cost justified by the performance gains, if any?

### Conclusion

Through this systematic experimentation and discussion, you'll gain insights into how different architectural choices for Vision Transformers affect model performance, especially in the context of a relatively simple dataset like MNIST. These findings can guide the design of transformer models for other tasks and datasets, balancing model complexity and performance.

# Artificial Neural Networks and Deep Learning

## 4. Genrative Models

## Section 4.1.1 : Energy-Based Models: Restricted Boltzmann Machines
### Q1. In the restricted boltzmann machine (RBM) script, the training algorithm refers to the pseudo-likelihood. Why is that? What is the consequence regarding the training of the model?
### Section 4.1.1 : Energy-Based Models: Restricted Boltzmann Machines

#### Q1. In the restricted Boltzmann machine (RBM) script, the training algorithm refers to the pseudo-likelihood. Why is that? What is the consequence regarding the training of the model?

1. **Intractable Likelihood Calculation**:
    - The goal of RBM training is to maximize the log-likelihood of the training data, ahcieved by learning over the gradient of the log-likelihood, or logrithmic of joint distribution $P_{model}(v, h; \theta)$. 
    - The exact maximum log-likelihood, characterized by an energy function $ E(v, h; \theta) = -v^TWh - b^Tv - a^Th $ defining the joint configuration of visible units $ v $ and hidden units $ h $ and a partition function $ Z(\theta) = \sum_v \sum_h \exp(-E(v, h; \theta)) $ denoting the sum of the exponentiated negative energies over all possible configurations of the visible and hidden units, involves calculating the derivatives of the logrithmic of joint distribution $P_{model}(v, h; \theta) = \frac{1}{Z(\theta)} \exp(-E(v, h; \theta)) $ that is interperted as the expectation difference between data-dependent expection with respect to data distribution $P_{data}(h, v; \theta)$ and model's expectation $P_{model}(v, h; \theta)$. 
    - However, this exact maximum likelihood learning is intractable for large models owing the exponentially growing number of terms.
2. **Solution**: 
    - To circumvent this, the Contrastive Divergence (CD) algorithm with conditional distributions, or conditional probabilities, factorized as
      $
      P(h|v; \theta) = \prod_j p(h_j|v)
      $
      $
      P(v|h; \theta) = \prod_i p(v_i|h)
      $
    using the sigmoid activation function, is used as an approximatoin, where the update rule for the weight matrix $ W $ is given by
    $
    \Delta W = \alpha (E_{P_{data}}[vh^T] - E_{P_T}[vh^T])
    $
    where $ P_T $ is the distribution obtained after running a Gibbs sampler for $ T $ steps.
    - This alternative objective function, known as the pseudo-likelihood, is used to approximate the log-likelihood of the data, and is defined as
    $
    \log P(v_i|v_{-i}; \theta) = \sum_i \log P(v_i|v_{-i}; \theta)
    $
    where $ v_{-i} $ denotes the visible units excluding the $ i $-th unit.
    
    - The pseudo-likelihood training algorithm simplifies the computation of the log-likelihood, avoiding the intractable calculation of the partition function $ Z(\theta) $, and focuses on the local dependencies between the visible units, rather than the global structure of the data. This allows for more efficient training, but at the cost of an approximation that might not capture all dependencies in the data as accurately as the true likelihood.

### Q2. What is the role of the number of components, learning rate and and number of iterations on the performance? You can also evaluate the effect it visually by reconstructing unseen test images.

#### Configuration Results:
- **Configuration 1**: n_components = 20, learning_rate = 0.01, n_iter = 10
- **Configuration 2**: n_components = 10, learning_rate = 0.01, n_iter = 10
- **Configuration 3**: n_components = 10, learning_rate = 0.001, n_iter = 10
- **Configuration 4**: n_components = 10, learning_rate = 0.001, n_iter = 30
- **Configuration 5**: n_components = 50, learning_rate = 0.001, n_iter = 30

1. **Effect of `n_components` (Number of Hidden Units)**:
   - **Configurations 1 and 2**: Increasing the number of hidden units from 10 to 20 (Configuration 2 to Configuration 1) significantly improves the pseudo-likelihood, indicating more hidden units allows for capturing more complex patterns in the data.
   - **Configuration 5**: Further increasing to 50 hidden units (Configuration 5) shows even better performance, suggesting that a higher number of hidden units can lead to better model capacity and improved performance.

2. **Effect of `learning_rate`**:
   - **Configurations 2 and 3**: A higher learning rate of 0.01 (Configuration 2) shows better performance compared to a lower learning rate of 0.001 (Configuration 3). The pseudo-likelihood decreases more significantly with a higher learning rate, indicating faster convergence with limited number of iterations. However, a higher learning rate may also lead to overshooting the optimal parameters.

3. **Effect of `n_iter` (Number of Iterations)**:
   - **Configurations 3 and 4**: Increasing the number of iterations from 10 to 30 (Configuration 3 to Configuration 4) improves the pseudo-likelihood slightly. This shows that more iterations allow the model more time to converge, but the improvement might be diminishing.
   - **Configuration 5**: With a larger number of hidden units, the number of iterations shows a significant impact, as seen in the continuous improvement in pseudo-likelihood even at 30 iterations.

The observations indicaes that the number of hidden units serve as the dominant factor in improving model performance, followed by the learning rate and number of iterations. While a lower learning rate in general may suggest avoidance in overshooting the optimal parameters, it is essential that such configuration would reqruie more training iteratoins allowing for the optimaml convergence. Without increment in number of hidden units, increment in number of iterations may not necessarily improve the model performance.

### Q3. Change the number of Gibbs sampling steps. Can you explain the result?

Gibbs sampling is a technique used in the Contrast Divergence (CD) algorithm delineated in $Q1$, where the model is trained by sampling from the model distribution using Markov Chain Monte Carlo (MCMC) methods that iteratively updates the states of the visible and hidden units to draw samples from the joint distribution. The number of Gibbs sampling steps determines the number of iterations the sampler runs to approximate the model distribution. 

With Gibb steps set to 50, the generated images are noisy such that the digits are vaguely recongnizable, indicating insufficient iterations to fully explore the state space.
With 100 steps, the quality of the generated images significantly improves, which result aligns with the learned expecation where increment in steps would results in a more comprehensive representation of data distribution. 
Finally, when validated with 200 steps,  distinct digits generated can be observed, indicating a closer approximation learned by the sampler to represent the true distribution. 

### Q4. Use the RBM to reconstruct missing parts of images. Discuss the results.
The RBM attempts to fill in the missing parts, characterized by variables `start_row_to_remove` and `end_row_to_remove`, based on the learned distribution across the number of Gibbs steps `reconstruction_gibbs_steps` from the training data. 

To evaluate the performance of RBM, the `end_row_to_remove` was initially set to `0` for emperically determining with which `number of gibbs steps`, or `reconstruction_gibbs_steps` can the RBM reconstruct the missing parts accurately, which `number of gibbs steps` was found to be `49`.

<p align="center">
<img src="image/rbm_reconstruction_step49.png" width="300" height="100">
<br>
<em>Fig.1 : Reconstructed digits with RBM gibbs steps of 49 </em>
</p>

Then, the `end_row_to_remove` was incremented stepwise to explore the limit of RBM's structure and patterns inference ability in given configuration of `100` hidden units, `0.01` learning rate, and `30` iterations. The model can accurately reconstruct the missing parts when `6` rows are removed and yet post `7` rows, the reconsutruction quality degraded, and at last completely failed to reconstruct when `20` rows were removed even with `number of gibbs steps` increased to `200`.

### Q5. What is the effect of removing ore rows in the image on the ability of the network to reconstruct? What if you remove rows on different locations (to, middle...)?

The reconstruction results are less likely to be affected if the removed sections are less critical to the overall shape of the digit. For a less affected removal instance owing to the still existence of the representative section with respect to the digit outline, four rows of removal in the top section for digits `7` or `9` would still allow a fair reconstruction as the lower parts contribute significantly to their shape. For digits such as `4`, `5`, or `6`, removing rows from the middle is often more detrimental since there exist a disruptive continuity of the digit's structure, leading to poorer RBM inference.

## Section 4.1.2: Energy-Based Models: Deep Boltzmann Machines

### Q1. load the pre-trainned Deep Boltzmann Machine (DBM) model that is trained on the MNIST dataset. Show the filters (interconnection weights) extracted from the previously trained RBM and the DBM, what is the difference? Can you explain why the difference between filters of the first and second layer of the DBM?
<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; text-align: center;">
    <img src="image/RBM_first_100_filters.png" width="190" />
    <figcaption>Fig.2 - RBM Filters</figcaption>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="image/DBM_first_100_filters.png" width="200" /> 
    <figcaption>Fig.3 - DBM First Layer Filters</figcaption>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="image/DBM_second_100_filters.png" width="200" /> 
    <figcaption>Fig.4 - DBM Second Layer Filters</figcaption>
  </div>
</div>

1. **Filters Extracted by the RBM and the 1st layer Filters Extracted by the DBM**:
    In grayscale, the filters in the DBM exhibit features in general with higher contrast. While the intensity of each pixel corresponds to the weight value, darker shades represent negative weights, lighter shades represent positive weights, and mid-gray represents weights close to zero, the DBM filters, with comparatively more darker shades exhibited in the filter indicating high negative weights in conjunction with high intensity features indicating high positive weights located in the filters, presents a significant potential in gradient descent optimization for edge detection compared to the generally-mid-gray RBM filters (Fig.2). Filters in the first layer of a DBM also capture basic features but with more variation and detail compared to RBM filters. 
2. **Filters Extracted by the 2nd Layer of the DBM**:
    Filters across different layers capture features at varying levels of abstraction owing to the hierarchical architecture of the model. Observed from Fig.3 and Fig.4, the first layer consist of filters with a largest shade cluster lighter than the second layer filters. This indicates the function of the first layer filters is to capture localized edge- and corner- like features in the region, which distinct patterns can be directly interpreted as parts of the input images, resembling stroke-like or edge-detecting features. 
    
    The second layer filters are more abstract and complex, capturing higher-order patterns in a global extend by integrating multiple low-level features. The second layer filters exhibit more intricate and abstract patterns, with darker shades, or negative wieghts covering the a specified round-shaped region, presumably where the digit distinctive outline is located at, with a contrastive light shades covering the reamining corner of the filters. This suggest a more complex broader area pattern learning and inter-local-feautres relationship learning are conducted at the second layer, essential for understanding the broader context and finer details within the data. Some filters in the second layer indicates a notable high intentisty circular small region in varied positions accross filters, which could be interpreted as the prominent feature detection of the digit's distinctive outline.

### Q2. Sample new images from the DBM. Is the quality better than the RBM from the previous exercise? Explain. 
The images from the DBM are of higher quality compared to the RBM due to the deeper architecture of the DBM, exhibiting highly distinguishable digits outline and notable digit prominent features that is clearer and more coherent compared to those generated by a single-layer RBM. The multi-layer structure of the DBM enables the learning of data hierarchical representations. According to the previous discussion that the first layer serves to captures low-level features such as edges and textures, while subsequent layers aims to capture higher-level abstractions and complex patterns, with more inter-assocaited intricated features learned across layers the DBM is destined to generate more realistic and detailed images compared to the single-layer, local dependency capturing RBM.

### Section 4.2: Generator and Discriminator in the Ring: Generative Adversarial Networks (GANs)
### Q1. Explain the different losses and results in the context of the GAN framework.

In the framework of Generative Adversarial Networks (GANs) framework comprises two neural networks: the generator and the discriminator. These networks are trained simultaneously in a two-player minimax game-theoretic, or zero-sum game, manner. Objectives of the generator and the discriminator netowrks competes against each other as adversaries. While the generator aims to, from the provided trainning data, produce fake and yet realistic data samples, whereas discriminator pose to distinguish between real data samples and those generated by the generator. As illustrated in the slides, the competing process facilitates a gammic zero-sum game implementatoin, referred to as minimax GAN, as the generator aims to minimize the value of value function $ v(\theta^G, \theta^D) $, while the discriminator aims to maximize the value function as its determined payoff, resulting in:

$ G^* = \arg \min_G \max_D v(G, D) $

In the default setting, the value function $ v(\theta^G, \theta^D) $ is expressed as:

$ v(\theta^G, \theta^D) = \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log (1 - D(G(z)))] $

To satisfy respective purposes in the traditional minimax game where the generator aims to minimize while the discriminator aims to maximize the value function, objective function for generator and discriminator are defined as follows:

$ J^{(G)}(G) = \mathbb{E}_{z \sim p_z} [\log (1 - D(G(z)))] $

$ J^{(D)}(D) = \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log (1 - D(G(z)))] $

It is worth highlighting that in practical terms, adjustments to mitigate issues of vanishing gradients imposed by $ D(x) $ reaching the equilibrium of the zero-sum game, referred to as the saddle points was made. Such that the generator objective was instead denoted as the non-saturating GAN objective function:

$ J^{(G)}(G) = -\mathbb{E}_{z \sim p_z} [\log D(G(z))] $

and the corresponding loss function for the generator is:

$ L_G = -\left( \mathbb{E}_{z \sim p_z} [\log D(G(z))] \right) $

Here, the generator tries to maximize $\log D(G(z))$, which is equivalent to minimizing $-\log D(G(z))$, such that $D(G(z))$ should be as large as possible, interpreted as that the discriminator will classify the generated data $G(z)$ as real with high probability. This modification aims to stabilize the training process by ensuring that the generator receives stronger gradients even when the discriminator's performance is good, addressing issues associated with vanishing gradients in the traditional minimax GAN.

The discriminator, on the other hand, aims to maximize the probability of correctly classifying real and fake data, such that the discriminator objective function is formulated as:

$ J^{(D)}(D) = \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log (1 - D(G(z)))] $

where the corresponding loss function for the discriminator is:

$ L_D = -\left( \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log (1 - D(G(z)))] \right) $

In the context of the algorithm from Goodfellow et al. (2014), the discriminator's loss can be broken down into two parts:
1. The loss for the real samples:
   $ L_{D,\text{real}} = -\mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] $
2. The loss for the fake samples:
   $ L_{D,\text{fake}} = -\mathbb{E}_{z \sim p_z} [\log (1 - D(G(z)))] $

Therefore, the total discriminator loss $ L_D $ is:
$ L_D = L_{D,\text{real}} + L_{D,\text{fake}} $

where both of $L_{D,\text{real}}$ and $L_{D,\text{fake}}$ should be minimized and approaching $0$ as the training progresses such that in the perspective of discriminator, $D(G(z))$ should be as close to zero as possible, indicating the generated data is classified as fake, and $D(x)$ should be as close to one as possible, indicating the real data is classified as real.

Reflected on the loss curve, the generator's loss shall increase over time, while the discriminator's loss shall decrease, indicating the generator's improving ability to deceive the discriminator, making it more challenging for the discriminator to differentiate between real and fake data as indicated in Fig.4. The training process is expected to progress smoothly, indicating a stable training process without significant oscillations or divergence, a common issue in GAN training.

After sufficient training, the generator distribution $ p_G $ becomes indistinguishable from the real data distribution $ p_{\text{data}} $. At this equilibrium, the discriminator cannot differentiate between real and fake samples, resulting in $ D(x) = \frac{1}{2} $ for any sample $ x $. One cannot further improve since $ p_{\text{G}} = p_{\text{data}} $, reaching plateau where the generator has learned to generate realistic data samples.

### Q2. What would you expect if the discriminator performs proportioanlly much better than the generator?
If the discriminator performs significantly better than the generator, several issues can arise, affecting the training dynamics and the quality of the generated data. The discriminator's performance can be considered too strong when it can easily distinguish between real and fake data with high confidence. This imbalance in performance indicates a calssification accurary with high confidence of $D(x) \approx 1$ for real data and $D(G(z)) \approx 0$ for generated data, leading to diminishing gradients for the generator. Suppose a batch of $m$ ,generator updates by decending its stochastic gradient:
$
  \nabla_{\theta^G} \frac{1}{m} \sum_{i=1}^m \log \left(1 - D(G(z^{(i)}))\right)
$

If $D(G(z)) \approx 0$, such vanishing gradients for the generator yield the gradients degrade, leading to the inability of generator to learn and improve, adverserally enforcing constant poor-quality samples generation.

### Q3. Discuss and illustrate the convergence and stability of GANs.

<p align="center">
<img src="image/GAN_loss_curve_generator_discriminator.png" width="200" height="150">
<br>
<em>Fig.4: GAN loss curve </em>
</p>

Configured with a `latent dimension` of `20`, a `batch size` of `512`, and a `learning rate` of `1e-3`, the empirical results detail the average losses for the generator and discriminator over `70` epochs. Key observations include:

#### Theoretical Illustration of GAN Convergence

1. **Adversarial Pair Near Convergence**:
   - $ p_G $ approximates $ p_{\text{data}} $, and the discriminator $ D $ achieves partial accuracy in classifying real versus fake data.

2. **Discriminator Training**:
   - The discriminator is optimized to distinguish between real and fake samples, converging to:
     $ D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)} $

3. **Generator Updates**:
   - The generator is updated based on the discriminator's feedback, producing samples $ G(z) $ that are increasingly realistic.

4. **Equilibrium State**:
   - At equilibrium, $ p_G \approx p_{\text{data}} $, and the discriminator's output for any sample $ x $ is:
     $ D(x) = \frac{1}{2} $

#### Empirical Training Process and Results

The provided GAN implementation includes the following key elements:

1. **Network Architecture**:
   - Generator: Fully connected layers with Leaky ReLU and Tanh activations.
   - Discriminator: Fully connected layers with Leaky ReLU and Dropout, followed by a sigmoid activation.

2. **Training Procedure**:
   - Discriminator updates:
     $
     \nabla_{\theta^D} \frac{1}{m} \sum_{i=1}^m \left[ \log D(x^{(i)}) + \log (1 - D(G(z^{(i)}))) \right]
     $
   - Generator updates:
     $
     \nabla_{\theta^G} \frac{1}{m} \sum_{i=1}^m \log (1 - D(G(z^{(i)})))
     $

#### Empirical Results

| Stage | Epoch | Average Generator Loss | Average Discriminator Loss |
|-------|-------|------------------------|----------------------------|
| Initial | 1 | 0.829 | 0.698 |
| Initial | 2 | 0.956 | 0.686 |
| Mid | 10 | 1.36 | 0.571 |
| Mid | 20 | 1.99 | 0.429 |
| Later | 30 | 2.22 | 0.396 |
| Later | 50 | 2.86 | 0.3 |
| Later | 70 | 3.08 | 0.253 |

In the context of convergence, both the generator and discriminator losses suggest a movement towards equilibrium corresponding to the theoretical expectation $ p_G \approx p_{\text{data}} $. The discriminator's loss decreases, showing improvement in classification, while the generator's loss increases but stabilizes, indicating that the generator is improving in producing realistic samples. Additionally, the smooth progression of both losses as indicated in Fig.4 without significant oscillations or divergence indicates a stable training process. Adjustments to learning rates and regularization techniques contribute to this stability.


### Q4.  Explore the latent space and discuss.
Three aspects of the latent space are explored: latent dimension, interpolation in latent space (traversing latent space), and the Sampling and Visualizing Multiple Latent Vectors.

Latent dimension was first explored empirically by comparing `latent_dim = 10` and `latent_dim = 20` with both `num_epochs = 70`, results tabulated below:

  | Latent Dimension | Epoch 1 Generator Loss | Epoch 1 Discriminator Loss | Epoch 25 Generator Loss | Epoch 25 Discriminator Loss | Epoch 50 Generator Loss | Epoch 50 Discriminator Loss |
  |------------------|------------------------|----------------------------|-------------------------|-----------------------------|-------------------------|-----------------------------|
  | 10               | 0.818                  | 0.705                      | 1.76                    | 0.463                       | 2.46                     | 0.354                       |
  | 20               | 0.829                  | 0.698                      | 2.13                    | 0.413                       | 2.86                     | 0.3                         |

This reveals that increment in latent dimension from `10` to `20` results in a higher generator loss, indicating that the generator is producing more realistic samples. The discriminator loss also decreases, suggesting that the discriminator is finding it more challenging to differentiate between real and fake data, which is a positive sign of the generator's improvement.

Interpolation in latent space, or discussed in the lecture as Traversing the latent space, was conducted by linearly interpolating between two latent vectors with parameter `lambda`. The generated images as illustrated in Fig.5, showing a smooth transition between number `5` and `3`. The interpolation indicates a semantically coherent and realistic morphing, or transition capability, of the generator of GAN in between latent vectors in the latent space.

<p align="center">
<img src="image/interpolate_the_latent_sapce.png" width="400" height="100">
<br>
<em>Fig.5: Latent vectors interpolation (GANs) </em>
</p>

Sampling multiple latent vectors, or discussed in the lecture as Sampling from the latent space, was conducted by randomly sampling multiple latent vectors and accordinly generating corresponding fake images. The generated images as illustrated in Fig.6, while showing the ability of generator producing relatively diverse and higher-quality samples and demostrating zero memory dependency on the training set for new image generation, latent code inference limitation of the GAN model constrained by imporsing focused on the models of the real-data distruction is inevitable indicated if compared to Variational Autoencoders (VAEs) or plain Autoencoders, which have an encoder.

<p align="center">
<img src="image/sample_latent_vector_from_prior_GAN_as_Generator.png" width="300" height="300">
<br>
<em>Fig.6: Latent vectors Sampling (GANs)</em>
</p>

### Q5. Try the CNN-based backbone and discuss.
CNN-based GANs, or Deep convolutional GAN (DCGAN), compared to the fully connected GANs, as disucessed in lecture, demonstrate a more realistic and diverse image generation capability well-suited to the application of Large-scale Scene Understanding. As DCGAN generator leverages a hierarchical structure of convolutional layers, it captures the distribution projected small spatial extent convolutional represetnations with feature maps, recognizing local patterns, textures, and dependenceis. Exploiting the aforementioned properties of convoltuional networks, the extracted feature maps are then upsampled, deconvoluted, expanding images from features to generate high-quality images. Conversely, the discriminator, performs an inverse operation, downsampling the input images to extract features, and classifying the images as real or fake.


### Q6. What are the advantages and disadvantgaes of GAN-models compared to other generative models, e.g. the auto-enocder family or diffusion models? Think about the conceptural aspects, the quality of the results, the training considerations, etc.

GANs with other prominent generative models, such as autoencoders and diffusion models, considering conceptual aspects, quality of results, and training considerations are addressed below:

The advantgaes of GANS, given the adversarial architecture such that the generator is implicitly guided by the discriminator, facilitates a realistic data generation by the discriminator-feedback-driven generator without explicitly modeling the data distribution.. This two-player zero-sum game nature of GAN model enables highly detailed and high-fidelity output which performance often surpass the quality of outputs from other generative models, especially when in conjunction with CNN-based backbone that the spatial pattern and texture learning advantages of convolutional layers are exploited, deriving a diversed architectures of GANs, such as DCGAN, Conditional GAN, Auxiliary Classifier GAN, InfoGAN, CycleGAN, StyleGAN, etc.

Howeber, disadvantages of GANs are also notable, such as the training instability resulting from vanishing gradients, resource-intensive yielding from database-distribution-based training and database-distribution-dependent inference nature requiring normal distributed large amounts of data. Additioanlly, as discussed in previous section the lack of underlying data distribution information and the intractablility of exact likelihood owing to the lack of data density estimation  makes it challenging to evaluate the likelihood of the generated samples.

Conversely, autoencoders, such as Variational Autoencoders (VAEs), characterized by amortized variational posterior, transfomring the problem into an Evidence Lower Bound optimization problem (ELBOW maximization), encodes the latent distribution in an encoder network and reconstruct the probalistic latent elements in a decoder network such that variations in the data distribution can be properly represented in the generated data. As the data variational distributions are modeled and parameterized for variational inference, VAEs provide a structured latent spac for utilizations, enabling a comparably stable training and more diversified sample generation stuiable for latent space interpolation and interpretable. However, VAEs often produce less detailed images compared to GANs. 

In the context of Diffuesion Models that generate data by reversing a diffusion process and gradually transforming noise into data samples, the series of iterative steps nature in the generation process, while promising in higher-fidelity which often surpass performance of GANs, require extensive computational resources and time for training. 

According to Prince 2023, "Generative models based on latent variables should habe the following properties: (1) Efficient sampling in terms of computationally inexpensive, (2) High-quality sampling in terms of indistinguishable from the real data, (3) Representative coverage in terms of sufficent coverage of the data distribution that the generated samples resembles a subset of the training samples, (4) Well-behaved latent space in terms of the latent space being continuous and interpretable, and (5) Disentagled latent space in terms of manipulation of the latent variables resulting in semantically meaningful and interpretable changes in the generated samples." The statment conlcudes the comparison of GANs, VAEs, and Diffusion Models, as demonstrated in the following table:

| Model      | Efficient | Sample Quality | Coverage | Well-behaved Latent Space | Disentangled Latent Space | Efficient Likelihood |
|------------|-----------|----------------|----------|---------------------------|---------------------------|----------------------|
| GANs       | ✓         | ✓              | ✗        | ✓                         | ?                         | n/a                  |
| VAEs       | ✓         | ✗              | ?        | ✓                         | ?                         | ✗                    |
| Flows      | ✓         | ✗              | ?        | ✓                         | ?                         | ✓                    |
| Diffusion  | ✗         | ✓              | ?        | ✗                         | ✗                         | ✗                    |

## Section 4.3: An Auto-Encoder with a Touch: Variational Auto-Encoders (VAEs)
### Q1. In practice, the model does not maximize the log-likelihood but another metric. Which one? Why is that and how does it work?

Variational Autoencoders (VAEs), instead of maximizing the log-likelihood which is computationally intractable for complex models, capatalize on parameterizing the variational distributions (latent variables) and approzimating which using Jensen's inequlaity: 

$ \ln p(x) = \ln \int p(x|z)p(z)dz \geq \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \ln \frac{p(x|z)p(z)}{q_\phi(z|x)} \right] $

Consifering an amortized variational posterior $ q_\phi(z|x) $ for each $x$, the VAE model maximizes the Evidence Lower Bound (ELBO) instead of the log-likelihood, which is defined as:

$ \ln p(x) \geq \text{ELBO} = \mathbb{E}_{z \sim q_\phi(z|x)}[\ln p(x|z)] - \mathbb{E}_{z \sim q_\phi(z|x)}(\ln q_\phi(z|x) - \ln p(z)) $

Where:
- $\mathbb{E}_{z \sim q_\phi(z|x)}[\ln p(x|z)]$ is the expected log-likelihood of the data given the latent variables, corresponding to the reconstruction error term measuring the model's ability to reconstruct the input data.
- $\mathbb{E}_{z \sim q_\phi(z|x)}(\ln q_\phi(z|x) - \ln p(z))$ is the Kullback-Leibler (KL) divergence $\text{KL}(q_\phi(z|x) \| p(z))$ between the approximate posterior $q_\phi(z|x)$ and the prior $p(z)$, acting as a regularization term ensuring the approximate posterior is close to the prior.

The lower bound of the log-likelihood is refered to as Evidence Lower Bound (ELBO), which is maximized during training. The ELBO balances the reconstruction error and the regularization term, facilitating an accurate representation of the latent space while stimulating a generalized model construction. 

In implementation, the $q_\phi(z|x)$ can be reparameterized by sampling from a common distribution such as a Gaussian distribution, characterized by $\mu_\phi$ mean, $\sigma_\phi$ standard deviation, and $\sigma^2_\phi$ variance, enabling the backpropagation of gradients in stochastic gradient descent training for ELBO maximization.

### Q2. In particular, what similarities and differences do you see when compared with stacked auto-encoder from the previous assignment? What is the metric for the reconstruction error in each case?

Both VAEs and stacked autoencoders use an encoder-decoder architecture where the encoder maps input data to a lower-dimensional latent space while the decoder reconstructs the data from this latent space, and that both models aim to minimize the reconstruction error matching the original input. However, there are key differences in the objective functions and the latent space regularization between VAEs and stacked autoencoders.


| Feature                      | VAE                                                                                 | Stacked Autoencoder                                                         |
|------------------------------|-------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| **Architecture**             | Encoder-decoder                                                                    | Encoder-decoder                                                            |
| **Latent Space Regularization** | Probabilistic framework with latent space defined by a distribution (e.g., Gaussian). Encoder outputs mean and variance, with sampling from this distribution. Regularization via KL divergence. | Deterministic mapping from input to latent space without enforced distribution over latent variables. |
| **Objective Function**       | Maximizes the Evidence Lower Bound (ELBO):                                          | Minimizes reconstruction error (e.g., Mean Squared Error).                |
| **Reconstruction Error Metric** | Part of ELBO, measured as expected log-likelihood of data given latent variables: $\mathbb{E}_{z \sim q_\phi(z|x)}[\ln p(x|z)]$. Uses MSE for continuous data or binary cross-entropy for binary data. | Typically uses Mean Squared Error (MSE) loss for continuous data.          |
| **Mathematical Formula**     | ELBO: $\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\ln p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))$ | Reconstruction Error: $\mathcal{L}(x, \hat{x}) = \| x - \hat{x} \|^2$    |

#### Variational Autoencoders (VAEs):
The learning algorithm of VAEs is defined as follows: 
  **Learning algorithm** using the distributions $ q_\phi(z|x) = \mathcal{N}(z|\mu_\phi(x), \sigma^2_\phi(x)) $, $ p(z) = \mathcal{N}(z|0, I) $, $ p_\theta(x|z) = \text{Categorical}(x|\theta(z)) $:

  1. Considering a data set $ D = \{x_n\}_{n=1}^N $, take $ x_n $ and apply the encoder network to get $ \mu_\phi(x_n), \sigma^2_\phi(x_n) $.

  2. Calculate $ z_{\phi,n} = \mu_\phi(x_n) + \sigma_\phi(x_n) \odot \epsilon $, with $ \epsilon \sim \mathcal{N}(\epsilon|0, I) $.

  3. Apply the decoder network to $ z_{\phi,n} $ to get the probabilities $ \theta(z_{\phi,n}) $.

  4. Calculate the ELBO by plugging in $ x_n, z_{\phi,n}, \mu_\phi(x_n), \sigma^2_\phi(x_n) $, maximizing

  $
  \text{ELBO}(D; \theta, \phi) = \sum_{n=1}^N \left\{ \ln \text{Categorical}(x_n | \theta(z_{\phi,n})) + [\ln \mathcal{N}(z_{\phi,n}|\mu_\phi(x_n), \sigma^2_\phi(x_n)) + \ln \mathcal{N}(z_{\phi,n}|0, I)] \right\}
  $

1. **Latent Space Regularization**:
   - **Probabilistic Framework**: In VAEs, the encoder maps input $ x $ to parameters $ \mu(x) $ and $ \sigma(x)^2 $ of a Gaussian distribution $ q_\phi(z|x) $. The latent variable $ z $ is then sampled from this distribution during training, introducing randomness and regularization.
   - **KL Divergence**: A regularization term is added to the loss function to ensure the learned latent distribution $ q_\phi(z|x) $ is close to the prior distribution $ p(z) $ (usually a standard Gaussian). This is measured using the Kullback-Leibler (KL) divergence: $\text{KL}(q_\phi(z|x) \| p(z))$.

2. **Objective Function**:
   - **Evidence Lower Bound (ELBO)**: The VAE objective function aims to maximize the ELBO, which balances the reconstruction accuracy and the regularization of the latent space distribution.
     $
     \mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\ln p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))
     $
   - **Reconstruction Term**: The first term, $ \mathbb{E}_{q_\phi(z|x)}[\ln p_\theta(x|z)] $, ensures the reconstructed output $ \hat{x} $ is similar to the input $ x $. For continuous data, this is typically implemented using Mean Squared Error (MSE) loss: $ \mathbb{E}_{z \sim q_\phi(z|x)}[-\|x - \hat{x}\|^2] $. For binary data, binary cross-entropy loss is used: $ \mathbb{E}_{z \sim q_\phi(z|x)}[-(x \log \hat{x} + (1 - x) \log (1 - \hat{x}))] $. In case of images $ x \in \{0, 1, ..., 255\}^D $, one cannot use a normal distribution. One can take a categorical distribution $[Tomczak 2022]$:
   $ p_\theta(x|z) = \text{Categorical}(x|\theta(z)) $ using a Neural Network (NN) for $ \theta(z) = \text{softmax}(\text{NN}(z)) $. $ p_\theta(x|z) = \text{Categorical}(x|\theta(z)) $ using a Neural Network (NN) for $ \theta(z) = \text{softmax}(\text{NN}(z)) $.

   - **Regularization Term**: The KL divergence term penalizes the divergence of the latent space distribution from the prior.

3. **Reconstruction Error Metric**:
   - Measured as the expected log-likelihood of the data given the latent variables, typically using MSE for continuous data or binary cross-entropy for binary data.

#### Stacked Autoencoders:
1. **Latent Space Regularization**:
   - **Deterministic Mapping**: The encoder deterministically maps input $ x $ to a latent space $ z $ without enforcing a distribution over the latent variables. There is no probabilistic interpretation or regularization.

2. **Objective Function**:
   - **Reconstruction Error**: The objective is to minimize the reconstruction error, ensuring the output $ \hat{x} $ is as close as possible to the input $ x $.
     $
     \mathcal{L}(x, \hat{x}) = \| x - \hat{x} \|^2
     $
   - There is no additional term to regularize the latent space distribution.

3. **Reconstruction Error Metric**:
   - Typically measured using Mean Squared Error (MSE) for continuous data.

The key difference of metrics for reconstruction error between VAE and stacked autoencoders lies in the probabilistic framework and regularization of the latent space in VAEs, which is absent in stacked autoencoders. The probabilistic nature of VAEs allows for decomposing the error into reconstruction error, $\mathbb{E}_{q_\phi(z|x)}[\ln p_\theta(x|z)] = \sum_{n=1}^N \{ \ln \text{Categorical}(x_n | \theta(z_{\phi,n}))\} $, and KL divergence, $\text{KL}(q_\phi(z|x) \| p(z)) = \frac{1}{2} \sum_{n=1}^N \{ 1 + \ln (\sigma_{\phi,n}^2) - \mu_{\phi,n}^2 - \sigma_{\phi,n}^2 \}$, providing a more structured and interpretable latent space. In contrast, stacked autoencoders focus solely on minimizing the reconstruction error without probabilistic constraints, leading to a more deterministic latent space.

### Q3. Explore the latent space using the provided code and discuss what you observe.
With configuration of `batch_size = 3000`, `latent_dim = 600`, `middle_dim = 300`, `learning_rate = 1e-3`, and `max_epochs = 100`, the empirical results of the latent space explorations samples of the VAE model VAE model are illustrated in Fig.7, where as the interpolations in the latent space are depicted in Fig.8.

<p align="center">
<img src="image/VAE_fake_reconstructed_digits_latent_dim_600_mid_dim_300_epoch_100.png" width="300" height="300">
<br>
<em>Fig.7: Latent vectors Sampling (VAEs) </em>
</p>

Fig.7 illustrates the latent space explorations samples of the VAE model, showing a more diversed generated digits compared to the varaibility capability of GANs indicated in Fig.6 given the same MIST dataset, capturing a more holistic distribution of the latent space owing to the probabilistic nature of the VAE model. Nontheless, fidelity and level of details in the VAE generated digits are notably less intricate compared to the ones generated by GANs, as the VAE model focuses on the reconstruction of the input data rather than the discrimiator-guided new data samples generation as in GANs.

<p align="center">
<img src="image/VAE_interpolate_the_latent_space_latent_dim_600_mid_dim_300_epoch_100.png" width="400" height="100">
<br>
<em>Fig.8: Latent vectors interpolation (VAEs) </em>
</p>

Fig.8 demonstrate a strong coherence with high-level varaitions transformation between digits `3`, `5`, `6`, and almost `0`. The variance demonstrated indicates a well-represented latent space, where the VAE model is capable of generating continously changing data with two more level of variance than the ones of GANs while maintaining a semantically meaningful and interpretable changes inter-latent vectors in the latent space.


### Q4. Compare the generation mechanism to GANs. You may optionally want to consider similar backbones for a fair comparison. What are the advantages and disadvantages?

#### **Generation Mechanism:**

**Variational Autoencoders (VAEs):**

1. **Probabilistic Framework:**
   - VAEs use a probabilistic approach to model the data distribution.
   - The encoder maps the input $ x $ to a latent space $ z $ defined by a distribution, usually a Gaussian with parameters (mean $\mu$ and variance $\sigma^2$).
   - The latent variable $ z $ is sampled from this distribution.
   - The decoder then reconstructs the data $ \hat{x} $ from $ z $.

2. **Objective Function:**
   - VAEs maximize the Evidence Lower Bound (ELBO), which is composed of the reconstruction term and the KL divergence term.
     $
     \mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\ln p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))
     $

**Generative Adversarial Networks (GANs):**

1. **Adversarial Framework:**
   - GANs consist of two neural networks: a generator $ G $ and a discriminator $ D $.
   - The generator $ G $ maps a random noise vector $ z $ to the data space, generating fake samples.
   - The discriminator $ D $ tries to distinguish between real and fake samples.
   - The training is a minimax game where the generator tries to fool the discriminator, and the discriminator tries to correctly identify real versus fake samples.

2. **Objective Function:**
   - The objective function for GANs involves two loss functions: one for the discriminator and one for the generator.
   - The generator's objective function:
     $
     J^{(G)}(G) = -\mathbb{E}_{z \sim p_z} [\log D(G(z))]
     $
   - The discriminator's objective function:
     $
     J^{(D)}(D) = \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log (1 - D(G(z)))]
     $

#### **Advantages and Disadvantages:**

| Aspect | VAEs | GANs |
| --- | --- | --- |
| **Training Stability** | Generally more stable due to probabilistic framework. The loss function is well-defined and gradients are usually smooth. | Training can be unstable due to the adversarial nature. It often requires careful tuning of hyperparameters and may suffer from mode collapse. |
| **Latent Space Structure** | Provides a well-defined and interpretable latent space, thanks to the regularization term (KL divergence). This leads to a smoother and more continuous latent space. | The latent space may not be as well-structured or interpretable. There is no explicit regularization enforcing a distribution over the latent space. |
| **Sample Quality** | Generated samples might be of lower quality compared to GANs because the decoder needs to model the entire data distribution. | GANs often produce high-quality, sharp images because the generator is directly trained to fool the discriminator, which enforces realism in the samples. |
| **Mode Coverage** | VAEs tend to cover the entire data distribution, generating diverse samples, but might be blurry. | GANs might suffer from mode collapse, where the generator produces a limited variety of samples, focusing on modes that can easily fool the discriminator. |
| **Reconstruction Ability** | Good reconstruction ability due to the explicit reconstruction term in the loss function. | GANs do not focus on reconstruction; they focus on generating realistic samples that can fool the discriminator. |
| **Computational Efficiency** | Training is generally computationally efficient and straightforward due to the absence of an adversarial component. | Training can be computationally intensive due to the need to optimize two networks simultaneously and the potential instability. |

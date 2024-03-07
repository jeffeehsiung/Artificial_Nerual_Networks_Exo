# Artificial Neural Networks and Deep Learning

## 1. Exploring Neural Networks for Regression and Classification

### Objective

In this lab session, we delve into the mechanisms of neural networks for regression and classification tasks, emphasizing training methods, generalization, and optimization algorithms which concepts are presented in section 1.3 "In the Wide Jungle of the Training Algorithms" and 1.4 "A Personal Regression Exercise."

### Introduction

This report investigates various gradient-based optimization algorithms and their impact on the training and generalization of neural models. Additionally, we present a hands-on experiment in approximating a nonlinear function using a neural network tailored to a unique dataset, highlighting the intricacies of model selection and evaluation.

### Methodology

The first part of our study scrutinizes different optimization algorithms using PyTorch implementations. We examine the role of noise in optimization processes, compare vanilla gradient descent with its stochastic and accelerated variants, assess the influence of network size on optimizer selection, and distinguish between epochs and time for algorithm speed evaluation.

In the second part, we tackle a regression problem, approximating an unknown nonlinear function based on a given dataset of 13,600 datapoints. Our approach involves constructing a personal dataset from five nonlinear functions, designing a neural network architecture, and evaluating its performance on a separate test set.

#### Section 1.3: A small model for a small dataset

- **Answer with plots and tabular numerical data**

**Q1. What is the impact of the noise parameter in the example with respect to the optimization process?**

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
- **Training Loss Plot:** We would expect to see potentially more fluctuations in the training loss over epochs with higher noise levels. The convergence might also be slower or less smooth compared to training on less noisy data.
- **Predicted Surface vs. True Surface:** The predicted surface plot would likely show more deviation from the true underlying function due to the noise. This deviation would manifest as a less smooth surface or one that doesn't capture the true pattern as cleanly.
- **Squared Residuals:** The plot of squared residuals (the squared differences between predictions and actual values) would likely show higher values on average, indicating greater prediction error due to the noise.

### Experimental Result

1. **Training and Validation Loss**
   - Description: Track the training and validation loss over epochs. An increasing gap between training and validation loss might indicate overfitting, which can be exacerbated by noise.
   - Quantification: Calculate the difference or ratio between training and validation loss. A larger difference suggests that noise may be negatively impacting generalization.

2. **Learning Curves**
   - Description: Plot learning curves by graphing the model's performance on the training and validation datasets over time.
   - Quantification: Analyze the slope of the learning curves. Fluctuations or plateaus in validation performance can indicate sensitivity to noise.

3. **Model Accuracy**
   - Description: Evaluate the model's accuracy (or other relevant metrics) on both the training set and an unseen test set.

**Q2. How does (vanilla) gradient descent compare with respect to its stochastic and accelerated versions?**

## Vanilla Gradient Descent

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

In contrast, **Stochastic Gradient Descent (SGD)** and **Accelerated versions** (such as Momentum, Nesterov Accelerated Gradient, Adam, etc.) introduce various optimizations to improve the convergence speed, efficiency, or stability of the training process. For example, SGD updates the model parameters using the gradient computed from a randomly selected subset of the data (a mini-batch) rather than the entire dataset, significantly speeding up the computation and allowing for more frequent updates.

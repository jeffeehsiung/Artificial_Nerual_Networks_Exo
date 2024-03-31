# Artificial Neural Networks and Deep Learning

## 3. Deep Feature Learning 

## Section 3.1: Autoencoders and Stacked Autoencoders

### Q1. Conduct image reconstruction on synthetic handwritten digits dataset (MNIST) using an autoencoder. Note that you can tune the number of neurons in the hidden layer (encoding dim) of the autoencoder and the number of training epochs (n epochs) so as to obtain good reconstruction results. Can you improve the performance of the given model?.

Using a separate validation set for parameter tuning and reserving the test set for final evaluation ensures an unbiased performance measure on unseen data. In the process, training data is split into a new training set (80%) and a validation set (20%), with the latter used for hyperparameter tuning. The model undergoes training on this adjusted training set, validation against the validation set, and is ultimately evaluated on the test set to assess its generalization capability.

### Q2. Conduct image classification on MNIST using an stacked autoencoder. Are you able to obtain a better result by changing the size of the network architecture? What are the results before and after fine-tuning? What is the benefit of pretraining the network layer by layer?

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

### Q2. The file cnn.ipynb runs a small CNN on the handwritten digits dataset (MNIST). Use this script to investigate some CNN architectures. Try out different amounts of layers, combinations of different kinds of layers, number of filters and kernel sizes. Note that emphasis is not on experimenting with batch size or epochs, but on parameters specific to CNNs. Pay close attention when adjusting the parameters for a convolutional layer as the dimensions of the input and output between layers must align. Discuss your results. Please remember that some architectures will take a long time to train.

## Section 3.3: Self-Attention and Transformers
### Q1. Please run both the NumPy and PyTorch implementations of the self-attention mechanism. Can you explain briefly how the dimensions between the queries, keys and values, attention scores and attention outputs are related? What do the query, key and value vectors represent? Note that the attention mechanism will also be discussed in lecture 11.

### Q2. Please train the Transformer on the MNIST dataset. You can try to change the architecture by tuning dim, depth, heads, mlp dim for better results. You can try to increase or decrease the network size and see whether it will influence the prediction results much. Note that ViT can easily overfit on small datasets due to its large capacity. Discuss your results under different architecture sizes.
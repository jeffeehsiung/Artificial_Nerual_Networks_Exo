# Artificial Neural Networks and Deep Learning

## 4. Genrative Models

## Section 4.1.1 : Energy-Based Models: Restricted Boltzmann Machines
### Q1. In the restricted boltzmann machine (RBM) script, the training algorithm refers to the pseudo-likelihood. Why is that? What is the consequence regarding the training of the model?
### Section 4.1.1 : Energy-Based Models: Restricted Boltzmann Machines

#### Q1. In the restricted Boltzmann machine (RBM) script, the training algorithm refers to the pseudo-likelihood. Why is that? What is the consequence regarding the training of the model?

1. **Intractable Likelihood Calculation**:
   - **Problem**: The goal of RBM training is to maximize the log-likelihood of the training data, ahcieved by learning over the gradient of the log-likelihood, or logrithmic of joint distribution $P_{model}(v, h; \theta)$. 
   
   The exact maximum log-likelihood, characterized by an energy function $ E(v, h; \theta) = -v^TWh - b^Tv - a^Th $ defining the joint configuration of visible units $ v $ and hidden units $ h $ and a partition function $ Z(\theta) = \sum_v \sum_h \exp(-E(v, h; \theta)) $ denoting the sum of the exponentiated negative energies over all possible configurations of the visible and hidden units, involves calculating the derivatives of the logrithmic of joint distribution $P_{model}(v, h; \theta) = \frac{1}{Z(\theta)} \exp(-E(v, h; \theta)) $ that is interperted as the expectation difference between data-dependent expection with respect to data distribution $P_{data}(h, v; \theta)$ and model's expectation $P_{model}(v, h; \theta)$. 
   
   However, this exact maximum likelihood learning is intractable for large models owing the exponentially growing number of terms.
   
   - **Solution**: To circumvent this, the Contrastive Divergence (CD) algorithm with conditional distributions, or conditional probabilities, factorized as
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
    
    This alternative objective function, known as the pseudo-likelihood, is used to approximate the log-likelihood of the data, and is defined as
    $
    \log P(v_i|v_{-i}; \theta) = \sum_i \log P(v_i|v_{-i}; \theta)
    $
    where $ v_{-i} $ denotes the visible units excluding the $ i $-th unit.
    
    - **Consequence**: The pseudo-likelihood training algorithm simplifies the computation of the log-likelihood, avoiding the intractable calculation of the partition function $ Z(\theta) $, and focuses on the local dependencies between the visible units, rather than the global structure of the data. This allows for more efficient training, but at the cost of an approximation that might not capture all dependencies in the data as accurately as the true likelihood.

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
<img src="rbm_reconstruction_step49.png" width="300" height="100">
<br>
<em>Figure: Reconstructed digits with RBM gibbs steps of 49 </em>
</p>

Then, the `end_row_to_remove` was incremented stepwise to explore the limit of RBM's structure and patterns inference ability in given configuration of `100` hidden units, `0.01` learning rate, and `30` iterations. The model can accurately reconstruct the missing parts when `6` rows are removed and yet post `7` rows, the reconsutruction quality degraded, and at last completely failed to reconstruct when `20` rows were removed even with `number of gibbs steps` increased to `200`.

### Q5. What is the effect of removing ore rows in the image on the ability of the network to reconstruct? What if you remove rows on different locations (to, middle...)?

The reconstruction results are less likely to be affected if the removed sections are less critical to the overall shape of the digit. For a less affected removal instance owing to the still existence of the representative section with respect to the digit outline, four rows of removal in the top section for digits `7` or `9` would still allow a fair reconstruction as the lower parts contribute significantly to their shape. For digits such as `4`, `5`, or `6`, removing rows from the middle is often more detrimental since there exist a disruptive continuity of the digit's structure, leading to poorer RBM inference.

## Section 4.1.2: Energy-Based Models: Deep Boltzmann Machines

### Q1. load the pre-trainned Deep Boltzmann Machine (DBM) model that is trained on the MNIST dataset. Show the filters (interconnection weights) extracted from the previously trained RBM and the DBM, what is the difference? Can you explain why the difference between filters of the first and second layer of  the DBM?

### Q2. Sample new images from the DBM. Is the quality better than the RBM from the previous exercise? Explain. 

### Interpretation of Results:


## Section 4.2: Generator and Discriminator in the RING: Generative Adversarial Ntworks (GANs)
### Q1. Explain the different losses and results in the context of the GAN framework.
### Q2. What would you expect if the discriminator performs proportioanlly much better than the generator?
### Q3. Discuss and illustrate the convergence and stability of GANs.
### Q4.  Explore the latent space and discuss.
### Q5. Try the CNN-based backbone and discuss.
### Q6. What are the advantages and disadvantgaes of GAN-models compared to other generative models, e.g. the auto-enocder family or diffusion models? Think about the conceptural aspects, the quality of the results, the training considerations, etc.

## Section 4.3: An Auto-Encoder with a Touch: Variational Auto-Encoders (VAEs)
### Q1. In practice, the model does not maximize the log-likelihood but another metric. Which one? Why is that and how does it work?
### Q2. In particular, what similarities and differences do you see when compared with stacked auto-encoder from the previous assignment? What is the metric for the reconstruction error in each case?
### Q3. Explore the latent space using the provided code and discuss what you observe.
### Q4. Compare the generation mechanism to GANs. You may optionally want to consider similar backbones for a fair comparison. What are the advantages and disadvantages?

## Section 4.4: (Optional) Generation and Creativity
- In this section, the goal is to create a practical generative model from scratch. Don’t hesitate to base yourself on the codes given in the previous sections. Choose a dataset of your liking, a model and train it.
- Discuss all your design and training choices thoroughly as well as the obtained results.

## Discussion Points

## Conclusion

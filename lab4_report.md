# Artificial Neural Networks and Deep Learning

## 4. Genrative Models

## Section 4.1.1 : Energy-Based Models: Restricted Boltzmann Machines
### Q1. In the restricted boltzmann machine (RBM) script, the training algorithm refers to the pseudo-likelihood. Why is that? What is the consequence regarding the training of the model?

### Q2. What is the role of the number of components, learning rate and and number of iterations on the performance? You can also evaluate the effect it visually by reconstructing unseen test images.

### Q3. Change the number of Gibbs sampling steps. Can you explain the result?

### Q4. Use the RBM to reconstruct missing parts of images.  Discuss the results.

### Q5. What is the effect of removing ore rows in the image on the ability of the network to reconstruct? What if you remove rows on different locations (to, middle...)?

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

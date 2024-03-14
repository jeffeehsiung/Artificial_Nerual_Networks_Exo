# Artificial Neural Networks and Deep Learning

## 2. Recurrent Nerual Networks

### Objective


## Section 2.1: Hopfield Network

### Q1. What is the impact of the noise parameter in the example with respect to the optimization process?

The actual behavior of the Hopfield network might lead to attractors different from the target patterns due to several reasons, including the capacity of the network, initial states of the neurons, and the presence of spurious states or local minima in the energy landscape. These spurious states are usually a result of the limited capacity of the Hopfield network and the nature of the update rule (synchronous or asynchronous). Typically, a Hopfield network can reliably store up to 0.15N patterns where N is the number of neurons.

**Random Inputs: Attractors vs. Targets**
From the random inputs, you can see that the final states often converge to one of the target patterns (`[1, 1]`, `[-1, -1]`, and `[1, -1]`). In some cases, the network converges to `[-1, 1]`, which is not one of the original target patterns. This state is known as a spurious state or a false attractor, which is a byproduct of the network's dynamics. It is a result of the symmetric nature of the weights in the Hopfield network, which inherently supports both the target pattern and its inverse as stable states.

**Symmetric Inputs: Attractors vs. Targets**
The symmetric inputs give interesting results. The final states like `[1, 0]`, `[0, 1]`, `[-1, 0]`, and `[0, -1]` are not exactly the target patterns but are somewhat between them. They represent states of high symmetry and are sometimes referred to as points of unstable equilibrium in the Hopfield network. The network does not converge to a specific attractor but rather to a state that is equidistant from multiple attractors. This is especially noticeable in the last case, where the final state `[0, 0]` is exactly in the middle of all the attractors and is essentially an unstable point.

#### To interpret these results:

1. **Convergence**: The network typically converges to one of the target patterns or their inverses. In the case of symmetric inputs, the convergence is to states of high symmetry, which are close to multiple attractors.

2. **Stability**: The true target patterns `[1, 1]` and `[-1, -1]` seem to be stable since they appear as final states in the simulations with random inputs. The pattern `[1, -1]` is also a stable state, while `[-1, 1]` is a spurious attractor. For symmetric inputs, the network does not converge to the target patterns but rather to intermediate states that are not stable attractors.

3. **Attractors**: The network does not always converge to the expected attractors (the targets used to create the network). This is due to the nature of Hopfield networks, where the energy landscape can have multiple minima, leading to these additional spurious attractors.

4. **Unwanted Attractors**: The presence of unwanted attractors such as `[-1, 1]` arises from the interactions between the neurons in the network. The network's energy function has local minima at these points, which are not explicitly trained for but are emergent properties of the network.

5. **Iterations**: The number of iterations to reach an attractor varies depending on the initial state and the network's energy landscape. For some initial states, convergence may be quick, while for others, especially those starting near the decision boundary between basins of attraction, it may take longer.

6. **Stability of Attractors**: The attractors `[1, 1]`, `[-1, -1]`, and `[1, -1]` are stable as they are the intended patterns that the network was trained on. The spurious attractor `[-1, 1]` is also stable, although it is not a desired state. The symmetric states are not stable attractors; they are more like saddle points in the energy landscape.
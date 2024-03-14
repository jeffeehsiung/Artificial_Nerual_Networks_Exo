# Artificial Neural Networks and Deep Learning

## 2. Recurrent Nerual Networks

## Section 2.1: Hopfield Network

### Q1. Hopfield network with target patterns [1, 1], [−1, −1] and [1, −1] and the corresponding number of neurons. Simulate finding attractors after a sufficient number of iterations for both Random and High-symmetry input vectors. Evaluate the obtained attractors.

1. **Attractor values**
- **Random Inputs: Attractors vs. Targets**
   <p align="center">
   <img src="random_inputs.png" width="300" height="200">
   <br>
   <em>Figure: Time evolution in the state space of random input.</em>
   </p>

    - The final states of the random input vectors converge to one of the target patterns (`[1, 1]`, `[-1, -1]`, and `[1, -1]`). 
    - In some cases, the network converges to `[-1, 1]`, which is not one of the original target patterns. This state is known as a spurious state or a false attractor, which is a byproduct of the network's dynamics. It is a result of the symmetric nature of the weights in the Hopfield network, which inherently supports both the target pattern and its inverse as stable states.

    <p align="center">
    <img src="energy_convolution_random.png" width="300" height="200">
    <br>
    <em>Figure: Energy evolution of the state of random input.</em>
    </p>

- **Symmetric Inputs: Attractors vs. Targets**

   <p align="center">
   <img src="symmetric_inputs.png" width="300" height="200">
   <br>
   <em>Figure: Time evolution in the state space of symmetric input.</em>
   </p>
    - The final states for symmetric inputs e.g. `[1, 0]`, `[0, 1]`, `[-1, 0]`, and `[0, -1]` are not among the orignal target patterns but ranges between them. The attractors represent states of high symmetry and are sometimes referred to as points of unstable equilibrium in the Hopfield network. 
    - The network does not converge to a specific attractor but rather to a state that is equidistant from multiple attractors. This is especially noticeable in the last case, where the final state `[0, 0]` is exactly in the middle of all the attractors and is essentially an unstable point.
- **Convergence**: 
    - The network typically converges to one of the target patterns or their inverses. In the case of symmetric inputs, the convergence is to states of high symmetry, which are close to multiple attractors.

    <p align="center">
    <img src="energy_convolution_symmetric.png" width="300" height="200">
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
   <img src="random_inputs_3D.png" width="300" height="200">
   <br>
   <em>Figure: Time evolution in the state space of 3D random input.</em>
   </p>

    - **Final State vs. Target Patterns**: The network converges to states that are either directly one of the target patterns or closely related. This indicates the network's ability to recall the stored patterns from various initial states.
    - **Stability and Recall Accuracy**: The stable final states ([1, -1, -1], [-1, -1, 1], and [1, 1, 1]) match the target patterns, demonstrating the network's associative memory properties. It suggests that these patterns are stable attractors in the network's energy landscape.
    - **Variability in Convergence**: Different initial states lead to convergence to different target patterns, showcasing the network's sensitivity to initial conditions and its capacity to differentiate between distinct attractors.

    <p align="center">
    <img src="energy_convolution_random_3D.png" width="300" height="200">
    <br>
    <em>Figure: Energy evolution of the state of 3D random input.</em>
    </p>

- **Symmetric Inputs: Attractors vs. Targets**
   <p align="center">
   <img src="symmetric_inputs_3D.png" width="300" height="200">
   <br>
   <em>Figure: Time evolution in the state space of 3D symmetric input.</em>
   </p>

    - **Near-symmetric States**: The final states for symmetric inputs often don't match exactly with the target patterns. Instead, they are near states of high symmetry, indicating that these inputs are near the boundary regions between the basins of attraction of the target patterns. This phenomenon illustrates the concept of "energy landscape" in Hopfield networks, where certain initial states can lead the network to converge to intermediate states close to multiple attractors.
    - **Stability of Symmetric States**: The appearance of states like [1, 0.06045472, -0.06045472] suggests that these symmetric or near-symmetric inputs do not strongly converge to one specific target pattern but rather to a state influenced by the surrounding attractors. This is particularly notable in a high-dimensional space, where the energy landscape can be complex.

    <p align="center">
    <img src="energy_convolution_symmetric_3D.png" width="300" height="200">
    <br>
    <em>Figure: Energy evolution of the state of 3D symmetric input.</em>
    </p>


2. **Average Number of Iterations to Reach an Attractor**
    - **Rapid Convergence for Random Inputs**: An average of 10 iterations to reach an attractor for random inputs suggests that the network can quickly stabilize to a memorized pattern from a variety of starting points.
    - **Immediate Convergence for Symmetric Inputs**: The reported average of 0 iterations for symmetric inputs might be misleading or an artifact of how convergence was measured. It likely indicates that these inputs are already very close to or within the basin of attraction of their final states from the beginning.

3. **Overall Interpretation**
    - **Existence of Spurious States**: The appearance of states not directly matching the targets, especially in symmetric simulations, points to the existence of spurious states or mixed states due to the complex interplay of attractors in the network's configuration space.

    In practical terms, these results illustrate the Hopfield network's capabilities and limitations as a content-addressable memory system, its sensitivity to initial states, and the influence of the network's structure on its dynamic behavior.
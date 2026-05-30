# Assignment #7.1: Adversarial Attacks on MNIST Classifier

## 📌 Presentation Overview (5-Minute Limit)
This project evaluates the adversarial vulnerability of our trained MNIST CNN. By implementing three white-box pixel perturbation algorithms (**FGSM**, **PGD**, and **Momentum IFGSM**), we analyze how imperceptible noise breaks neural network representations and compare their respective Adversarial Success Rates (ASR).

---

## 🧪 Implemented Attack Methodologies

All attacks operate in a **white-box setting**, leveraging direct access to the model's gradients (nabla_x L(theta, x, y)$) under strict $L_\infty$ or $L_2$ budget constraints ($\epsilon$).

### 1. Fast Gradient Sign Method (FGSM)
* **Type:** Single-step fast optimization.
* **Mechanism:** Computes the gradient of the loss with respect to the input image and takes a single step in the direction of the sign of the gradient.
* $$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(\theta, x, y))$$

### 2. Projected Gradient Descent (PGD)
* **Type:** Iterative optimization (Multi-step FGSM).
* **Mechanism:** Takes multiple smaller gradient steps ($\alpha$) and projects the resulting perturbation back into the feasible $\epsilon$-ball after each iteration. It acts as an iterative local optimizer to find the worst-case noise.
* $$x^{t+1} = \Pi_{x + \mathcal{S}} \left( x^t + \alpha \cdot \text{sign}(\nabla_x L(\theta, x^t, y)) \right)$$

### 3. Momentum Iterative FGSM (MI-FGSM)
* **Type:** Momentum-integrated iterative optimization.
* **Mechanism:** Accumulates velocity vectors into the gradient updates across iterations. This stabilizes the update directions, helping the attack escape poor local maxima and plateau regions.
* $$g_{t+1} = \mu \cdot g_t + \frac{\nabla_x L(\theta, x^t, y)}{\|\nabla_x L(\theta, x^t, y)\|_1}$$
* $$x^{t+1} = \Pi_{x + \mathcal{S}} \left( x^t + \alpha \cdot \text{sign}(g_{t+1}) \right)$$

---

## 📊 Comparative Evaluation Matrix
The metrics below demonstrate the baseline model performance versus model degradation under adversarial stress:

| Metric | Clean Baseline | FGSM Attack | PGD Attack | Momentum IFGSM |
| :--- | :---: | :---: | :---: | :---: |
| **Model Accuracy** | [99.21 %] | [76.47 %] | [99.99 %] | [100 %] |

*(Note: Data automatically extracted from local execution log `results.txt`)*

---

## 💡 Key Analytical Findings
1.  **Iterative Dominance:** Iterative attacks (**PGD** and **MI-FGSM**) yield a vastly superior ASR compared to single-step **FGSM**, proving that linear approximations fail to fully capture the complex decision boundaries of the CNN.
2.  **Momentum Efficiency:** Momentum IFGSM maintains effective optimization paths even in flat loss landscapes, making it the most lethal threat vector against this model structure.
3.  **Conclusion:** High test accuracy on a clean dataset creates a false sense of security; the model remains highly fragile to structured, gradient-directed adversarial input.

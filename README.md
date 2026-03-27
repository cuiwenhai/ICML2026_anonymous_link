## 1. Empirical Risk vs. $\lambda$

<div align="center">
  <img src="q1_p1.png" width="500">
</div>

> **Figure 1: Empirical Risk vs. $\lambda$.** This plot validates that our kernel-based approach achieves lower empirical risk and significantly reduced variance compared to the standard empirical method across varying interpolation parameters.

---

## 2. Empirical Unfairness vs. $\lambda$

<div align="center">
  <img src="q1_p2.png" width="500">
</div>

> **Figure 2: Empirical Unfairness vs. $\lambda$.** This figure compares the unfairness levels across 100 independent trials. Note the highly stable confidence bands achieved by the kernel smoothing technique.

## 3. Original Pareto Frontier and Optimal Trade-off Point

<div align="center">
  <img src="q4_p1.png" width="500">
</div>

> **Figure 3: Original Pareto Frontier.** This figure illustrates the empirical trade-off between predictive risk and unfairness un . The yellow marker indicates the optimal knee point ($\lambda=0.52$) identified by the Kneedle algorithm, representing the most cost-effective operating point for real-world deployment.

---

## 4. Normalized Pareto Frontier and Kneedle Mathematical Logic

<div align="center">
  <img src="q4_p2.png" width="500">
</div>

> **Figure 4: Normalized Pareto Frontier.** This figure demonstrates the underlying mathematical mechanism of the optimal $\lambda$ selection. By mapping the frontier into a $[0, 1]$ normalized space, the Kneedle algorithm rigorously identifies the equilibrium point that maximizes the vertical distance to the reference chord.

---

## 5. Trade-off Dynamics over Interpolation Parameter $\lambda$

<div align="center">
  <img src="q4_p3.png" width="500">
</div>

> **Figure 5: Trade-off Dynamics.** This figure plots the empirical risk and unfairness against the interpolation parameter $\lambda$. It empirically verifies our Theorems 3.5 and 3.6: as $\lambda$ increases, the unfairness scales strictly linearly while the empirical risk decreases monotonically. The red dashed line marks the  knee point  point.

## Fugure 1. Empirical Risk vs. $\lambda$

<div align="center">
  <img src="q1_p1.png" width="500">
</div>

> **Figure 1: Empirical Risk vs. $\lambda$.** This plot validates that our kernel-based approach achieves lower empirical risk and significantly reduced variance compared to the standard empirical method across varying interpolation parameters.

---

## Figure 2. Empirical Unfairness vs. $\lambda$

<div align="center">
  <img src="q1_p2.png" width="500">
</div>

> **Figure 2: Empirical Unfairness vs. $\lambda$.** This figure compares the unfairness levels across 100 independent trials. Note the highly stable confidence bands achieved by the kernel smoothing technique.

## Figure 3. Original Pareto Frontier and Optimal Trade-off Point

<div align="center">
  <img src="q4_p1.png" width="500">
</div>

> **Figure 3: Original Pareto Frontier.** This figure illustrates the empirical trade-off between predictive risk (LAD) and unfairness. The yellow marker indicates the optimal knee point ($\lambda=0.52$) identified by the Kneedle algorithm, representing the most cost-effective operating point for real-world deployment.

---

## Figure 4. Normalized Pareto Frontier

<div align="center">
  <img src="q4_p2.png" width="500">
</div>

> **Figure 4: Normalized Pareto Frontier.** This figure demonstrates the underlying mathematical mechanism of the optimal $\lambda$ selection. By mapping the frontier into a $[0, 1]$ normalized space, the Kneedle algorithm rigorously identifies the equilibrium point that maximizes the vertical distance to the reference chord.

---

## Figure5. Trade-off Dynamics over Interpolation Parameter $\lambda$

<div align="center">
  <img src="q4_p3.png" width="500">
</div>

> **Figure 5: Trade-off Dynamics.** This figure plots the empirical risk (LAD)  and unfairness against the interpolation parameter $\lambda$. It empirically verifies our Theorems 3.5 and 3.6: as $\lambda$ increases, the unfairness scales strictly linearly while the empirical risk decreases monotonically. The red dashed line marks the  knee point  point.



**Table 1: Performance Comparison of Empirical and Kernel-based CDF Estimators under Huber Loss (100 Trials).**

| Estimator | Empirical Risk (Mean) | Empirical Risk (Std Dev) | Empirical Unfairness (Mean) | Empirical Unfairness (Std Dev) |
| :--- | :--- | :--- | :--- | :--- |
| RDNN | 52.36271 | 1.7622 | 48.8390 | 6.4750 |
| FRWB | 57.1433 | 1.9206 | 1.7391 | 1.8635 |
| Ours with Empirical CDF | 56.6733 | 2.0153 | **1.7220** | 1.8318 |
| Ours with Kernel CDF | **56.2993** | **1.7380** | 2.1211 | **1.0883** |



**Table 2: The runtime (in seconds) with varying sample sizes $(n)$ and numbers of I-spline basis functions ($J _{n}$).**

| $n \setminus J _{n}$ | 6 | 9 | 12 | 15 | 18 | 23 | 28 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 500 | 0.0480 | 0.0960 | 0.2734 | 0.4559 | 0.8092 | 1.1964 | 0.5196 |
| 1000 | 0.0640 | 0.4188 | 0.3572 | 0.7667 | 0.3001 | 0.3851 | 0.8987 |
| 2000 | 0.1901 | 0.1685 | 1.3640 | 1.5791 | 0.3782 | 1.0496 | 2.6441 |
| 5000 | 0.2721 | 0.2382 | 1.8678 | 2.4900 | 2.6850 | 0.4334 | 3.6759 |
| 10000 | 0.3334 | 1.0428 | 1.9577 | 4.5505 | 3.2013 | 5.4520 | 5.4429 |
| 20000 | 0.5125 | 1.8884 | 4.3074 | 7.1253 | 9.2683 | 4.7845 | 9.6148 |

---

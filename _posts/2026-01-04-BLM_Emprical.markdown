---
layout: post
title: "Black Litterman Remarks"
date: 2026-01-04 09:00:00
tags: [research]
category: blog
hidden: false
---
## Some remarks about the black litterman model implementation 



###Absolute versus relative view:

A core modeling choice in the **Black–Litterman (BL)** framework is whether investor views are expressed in **absolute** or **relative** terms. This choice directly affects the structure of the **view matrix** $\mathbf{P}$, the **view vector** $\mathbf{Q}$, and ultimately the posterior expected returns.

An **absolute view** expresses a belief about the **level of return** of a single asset (or portfolio):

Assume three assets:
$$
\boldsymbol{\mu} =
\begin{bmatrix}
\mu_1 \\ \mu_2 \\ \mu_3
\end{bmatrix}
$$
**View:**

> Asset 1 will have an expected return of 6%.

$$
\mathbf{P} =
\begin{bmatrix}
1 & 0 & 0
\end{bmatrix},
\quad
\mathbf{Q} =
\begin{bmatrix}
0.06
\end{bmatrix}
$$

This encodes:
$$
\mu_1 = 6\%
$$

### Multiple Absolute Views

> Asset 1 → 6%
>  Asset 3 → 4%

$$
\mathbf{P} =
\begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix},
\quad
\mathbf{Q} =
\begin{bmatrix}
0.06 \\
0.04
\end{bmatrix}
$$

------

### 

A **relative view** expresses a belief about **performance differences** between assets:

> “Asset $i$ will outperform asset $j$ by $q$.”

------

### Example (Pairwise Relative View)

**View:**

> Asset 1 will outperform Asset 2 by 2%.

$$
\mathbf{P} =
\begin{bmatrix}
1 & -1 & 0
\end{bmatrix},
\quad
\mathbf{Q} =
\begin{bmatrix}
0.02
\end{bmatrix}
$$

This encodes:
$$
\mu_1 - \mu_2 = 2\%
$$

------

### Relative View vs Market Neutrality

Relative views **do not pin down levels**—only differences.
 This makes them:

- More **robust**
- Less sensitive to equilibrium assumptions
- Popular with **long–short** and **factor-based** portfolios

------

### Multi-Asset Relative View (Portfolio View)

> Technology (Assets 1 & 2) will outperform Energy (Asset 3) by 3%.

$$
\mathbf{P} =
\begin{bmatrix}
0.5 & 0.5 & -1
\end{bmatrix},
\quad
\mathbf{Q} =
\begin{bmatrix}
0.03
\end{bmatrix}
$$

This implies:
$$
0.5 \mu_1 + 0.5 \mu_2 - \mu_3 = 3\%
$$

### Mixed Absolute and Relative Views

In practice, **most BL implementations mix both**.
$$
\mathbf{P} =
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & -1
\end{bmatrix},
\quad
\mathbf{Q} =
\begin{bmatrix}
0.06 \\
0.02
\end{bmatrix}
$$
Meaning:

1. Asset 1 has 6% expected return
2. Asset 2 outperforms Asset 3 by 2%




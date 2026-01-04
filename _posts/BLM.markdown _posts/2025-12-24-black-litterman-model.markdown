---
title: "Black–Litterman Model 101"
layout: post
date: 2025-12-24 22:44
image: /assets/images/black.jpeg
headerImage: true
tags:
  - research
  - working-paper
star: true
category: blog
author: Xiaohu Zhang
description: Unwrapping the black box of the Black–Litterman model
---

## Black–Litterman Model: A Demystification

Two years ago, inspired by recent advances in 401(k) and pension fund investment strategies, I began exploring a research problem originally introduced nearly two decades ago by Fischer Black and Robert Litterman at Goldman Sachs.

Although the problem itself is well known, the mathematical intuition behind the Black–Litterman model—and, more importantly, how to apply it effectively in practice—is rarely explained in a clear and accessible way.

In this blog series, I walk through the Black–Litterman model from first principles, demystify the underlying mathematics, and show how it can be implemented in real-world portfolio construction. The goal is to bridge the gap between theory and practice and provide a practical guide beyond textbook formulas.

---

## Math Behind the Model

### 1. A Brief Review of Bayes’ Theorem

We begin with Bayes’ theorem:

$$
p(B \mid A) = \frac{p(A \mid B)\, p(B)}{p(A)}.
$$

This follows directly from the definition of joint probability:

$$
p(A, B) = p(A \mid B)\, p(B) = p(B \mid A)\, p(A).
$$

Dividing both sides by \( p(A) \) yields Bayes’ theorem.

Here, \( p(B) \) is the **prior** (marginal) probability of event \( B \), while \( p(B \mid A) \) is the **posterior** probability after observing event \( A \).

---

### 2. Bringing Data into Bayes’ Theorem

Bayesian statistics represents uncertainty about model parameters using probability distributions. A prior distribution reflects beliefs before observing data, which are updated using observed data to obtain a posterior distribution.

Bayes’ theorem in distributional form is:

$$
f(\theta \mid \text{data})
=
\frac{f(\text{data} \mid \theta)\, f(\theta)}{f(\text{data})}.
$$

Where:

- \( f(\theta \mid \text{data}) \): posterior distribution  
- \( f(\text{data} \mid \theta) \): sampling density (likelihood up to a constant)  
- \( f(\theta) \): prior distribution  
- \( f(\text{data}) \): marginal likelihood  

For a continuous parameter space:

$$
f(\text{data})
=
\int f(\text{data} \mid \theta)\, f(\theta)\, d\theta.
$$

This quantity—called the **marginal likelihood** or **evidence**—normalizes the posterior so it integrates to one.

Since it does not depend on \( \theta \), Bayes’ rule is often written in proportional form:

$$
f(\theta \mid \text{data}) \propto f(\text{data} \mid \theta)\, f(\theta).
$$

> **Posterior ∝ Likelihood × Prior**

---

## Bayesian Formulation with Financial Interpretation

To make the framework concrete, replace the abstract parameter \( \theta \) with financial quantities:

- \( \mu \): unknown expected return  
- \( r \): observed asset returns  

Bayes’ theorem becomes:

$$
p(\mu \mid r)
=
\frac{p(r \mid \mu)\, p(\mu)}{p(r)}.
$$

Where:

- \( p(\mu \mid r) \): posterior expected returns  
- \( p(r \mid \mu) \): likelihood  
- \( p(\mu) \): prior  
- \( p(r) \): marginal likelihood  

With:

$$
p(r) = \int p(r \mid \mu)\, p(\mu)\, d\mu.
$$

---

## Multivariate Normal Distribution

For portfolio analysis, the multivariate normal distribution is fundamental.

Let \( x \in \mathbb{R}^p \):

$$
f(x)
=
\frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}}
\exp\!\left(
-\frac{1}{2}(x - \mu)^{\mathsf T}\Sigma^{-1}(x - \mu)
\right),
$$

where:

- \( \mu \): mean vector  
- \( \Sigma \): covariance matrix  

---

## Multivariate Normal Likelihood with Bayesian Updating

Assume observed returns:

$$
r_1, \dots, r_n \in \mathbb{R}^p,
\qquad
r_i \mid \theta, \Sigma \sim \mathcal{N}(\theta, \Sigma).
$$

### Likelihood

$$
f(r_1,\dots,r_n \mid \theta, \Sigma)
=
(2\pi)^{-np/2} |\Sigma|^{-n/2}
\exp\!\left\{
-\frac{1}{2}
\sum_{i=1}^n
(r_i - \theta)^{\mathsf T}\Sigma^{-1}(r_i - \theta)
\right\}.
$$

Ignoring terms independent of \( \theta \):

$$
\theta \mid R, \Sigma
\sim
\mathcal{N}
\!\left(
\bar r,
\frac{1}{n}\Sigma
\right),
$$

where \( \bar r \) is the sample mean.

---

## The Black–Litterman Model
Now we can focus on implementing the Black-Litterman Model.

### 1. Return-Generating Process

$$
r_t \mid \mu, \Sigma \sim \mathcal{N}(\mu, \Sigma).
$$

This identifies:

$$
\theta \;\leftrightarrow\; \mu.
$$

---

### 2. Why Not Use Sample Means?

Classical Bayesian updating yields:

$$
\mu \mid \bar r
\sim
\mathcal{N}
\!\left(
\bar r,
\frac{1}{T}\Sigma
\right).
$$

Black–Litterman replaces historical averages with **market equilibrium and investor views**.

---

### 3. Prior: \( \pi \) and \( \tau\Sigma \)

#### Reverse Optimization

Mean–variance optimization:

$$
\max_w
\left(
w^{\mathsf T}\mu
-
\frac{\lambda}{2} w^{\mathsf T}\Sigma w
\right).
$$

First-order condition:

$$
\mu = \lambda \Sigma w.
$$

Setting \( w = w_{\text{mkt}} \):

$$
\pi = \lambda \Sigma w_{\text{mkt}}.
$$

Thus:

$$
\mu \sim \mathcal{N}(\pi, \tau\Sigma).
$$

---

### 4. Likelihood: Investor Views

$$
Q = P\mu + \varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0, \Omega).
$$

Replacing:

$$
\bar r \mid \mu \sim \mathcal{N}\!\left(\mu, \frac{1}{T}\Sigma\right)
$$

with:

$$
Q \mid \mu \sim \mathcal{N}(P\mu, \Omega).
$$

---

### 5. Posterior Distribution

$$
\mu \mid Q
\sim
\mathcal{N}(\mu_{\text{BL}}, \Sigma_{\text{BL}}),
$$

with:

$$
\mu_{\text{BL}}
=
\pi
+
\tau\Sigma P^{\mathsf T}
\left(
P\tau\Sigma P^{\mathsf T} + \Omega
\right)^{-1}
(Q - P\pi).
$$

---

## Summary

Black–Litterman replaces historical sample means with economically meaningful **soft observations**.

- \( \pi \): market equilibrium  
- \( P, Q \): investor views  
- \( \tau\Sigma, \Omega \): confidence levels  

Mathematically, it is simply **Bayesian updating under a Gaussian model**.
The Black–Litterman model can be fully understood as a direct application of Bayesian updating and Gaussian conditioning, and the references above provide both the theoretical foundation and practical insight needed to move from equations to implementation.
---

## References and Further Reading

1. **Ren-Raw Chen, Shih-Kuo Yeh, Xiaohu Zhang**  
   *On the Black–Litterman Model: Learning to Do Better*
    https://faculty.fordham.edu/rchen/JFDS-Chen.pdf
    
2. **Stephen Satchell**  
   *A Demystification of the Black–Litterman Model*  
   *Journal of Asset Management*, 2000

3. **Brian Junker**  
   *Basics of Bayesian Statistics*  
   https://www.stat.cmu.edu/~brian/463-663/week09/Chapter%2003.pdf

4. **Sayan Mukherjee**  
   *Useful Properties of the Multivariate Normal*  
   https://www2.stat.duke.edu/~sayan/Sta613/2018/lec/Bayesreg.pdf

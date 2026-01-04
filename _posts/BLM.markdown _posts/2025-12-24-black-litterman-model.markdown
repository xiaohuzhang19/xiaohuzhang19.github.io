---
title: "Black Litterman Model 101"
layout: post
date: 2025-12-24 22:44
image: /assets/images/black.jpeg
headerImage: false
tag:
- markdown
- elements
star: true
category: blog
author: Xiaohu Zhang
description: Unwrap the blackbox of Black Litterman Model 
---
### Black-Litterman Model **Demystification**

Two years ago, inspired by recent advances in 401(k) and pension fund investment strategies, I began exploring a research problem originally introduced nearly two decades ago by Fischer Black and Robert Litterman at Goldman Sachs.

Although the problem itself is well known, the mathematical intuition behind the Black–Litterman model—and, more importantly, how to apply it effectively in practice—is rarely explained in a clear and accessible way.

In this blog series, I will walk through the Black–Litterman model from first principles, demystify the underlying mathematics, and show how it can be implemented in real-world portfolio construction. The goal is to bridge the gap between theory and practice, and to provide a practical guide for practitioners who want to use the model beyond textbook formulas.

## Math behind the model

#### Intro to Bayes theorem

We start with the Bayes' theorem:
$$
p(B|A)=\frac{P(A|B)P(B)}{P(A)}
$$
The proof of bayes is relatively very straightforward.

​	Given:
$$
p(A,B)=p(A|B)p(B)
$$
​	Obviously, 
$$
p(B,A)=p(A,B)
$$
and we can state that 
$$
p(B,A)=p(B|A)p(A)
$$
set up two side equally, and divided by $p(A)$, we will have equaition (1)

 $p(B)$ is the unconditional (marginal) probability of the event of interest, also known as the *prior* information, is the marginal probability of event happening, not knowing anything beyond the fact W.R.T this event.

We may know some events or research tests i prior to the event B, which is the *posterior* information, can be denoted as $p(B|Tests_i...)$.

###### Bring data into Bayesian theorem

Now, if we bring probability distributions  into the theorem. 

*Bayesian statistics* views uncertainty about model parameters as something that can be explicitly quantified using probability distributions. A prior distribution represents initial beliefs about a parameter, which are then updated with observed data to produce a posterior distribution with reduced uncertainty. Under this framework, probability is interpreted as a measure of uncertainty rather than long-run frequency.

we can rewrite the bayes theorem in terms of distribution:
$$
f(\theta|data)=\frac{f(data|\theta)f(\theta)}{f(data)}
$$
where $f(\theta|data)$ is the posterior distribution for the parameter $\theta$, $f(data|\theta)$ is the sampling density for the data—which is proportional to the Likelihood function, only differing by a constant that makes it a proper density function—$f(\theta)$is the prior distribution for the parameter.  and $f(data)$ is the marginal probability of the data. 



For a continuous parameter space, the marginal probability of the data is given by
$$
f(\text{data}) = \int f(\text{data} \mid \theta)\, f(\theta)\, d\theta,
$$
which represents the integral of the sampling density multiplied by the prior distribution over all possible values of $\theta$.

This quantity is commonly referred to as the **marginal likelihood** (or **model evidence**). Its primary role is to normalize the posterior distribution so that it integrates to one. Although it often appears as a scaling constant, the marginal likelihood plays an important role in Bayesian model 

Because this denominator does not depend on the value of $\theta$ once the data are observed—and simply rescales the posterior—the Bayesian updating rule is often written in proportional form:
$$
f(\theta \mid \text{data}) \propto f(\text{data} \mid \theta)\, f(\theta)
$$
That is,

***Posterior ∝ Likelihood × Prior***



## Bayesian Formulation with Financial Interpretation

To make the Bayesian formulation more concrete, we can replace the abstract parameter $\theta$ with quantities commonly used in finance.

For example, let:

- $\mu$: unknown expected return
- $r$: observed asset returns

Bayes’ theorem can then be written as:

$p(μ∣r)=\frac{p(r)p(r∣μ)}{p(μ)}$

where:

- **$p(\mu \mid r)$** is the **posterior distribution** of the expected return,
- **$p(r \mid \mu)$** is the **likelihood**, describing how likely the observed returns are given $\mu$,
- **$p(\mu)$** is the **prior distribution**, encoding beliefs about expected returns before observing data,
- **$p(r)$** is the **marginal likelihood**, which serves as a normalization constant.

The **marginal likelihood** is defined as:
$$
p(r) = \int p(r \mid \mu)\, p(\mu)\, d\mu
$$


### Bring probability distribution into Bayesian: Bayesian Inference ###

Bayesian inference is typically carried out in the following steps:

1. **Prior specification**
    Choose a probability density $f(\theta)$, called the **prior distribution**, which represents our beliefs about the parameter $\theta$ before observing any data.
2. **Model specification**
    Specify a statistical model $f(\text{data} \mid \theta)$, which describes the distribution of the data conditional on the parameter $\theta$.
3. **Posterior updating**
    After observing the data, update prior beliefs and compute the **posterior distribution** $f(\theta \mid \text{data})$.



##### Posterior Distribution via Bayes’ Theorem

By Bayes’ theorem, the posterior distribution can be written as
$$
f(\theta \mid \text{data})
= \frac{f(\text{data} \mid \theta)\, f(\theta)}{f(\text{data})}
$$
or equivalently,
$$
f(\theta \mid \text{data})
= \frac{\mathcal{L}(\theta)\, f(\theta)}{c}
\propto \mathcal{L}(\theta)\, f(\theta),
$$
where:

- $\mathcal{L}(\theta) = f(\text{data} \mid \theta)$ is the **likelihood function**, viewed as a function of $\theta$;
- $c = f(\text{data})$ is the **normalizing constant**, also known as the **marginal likelihood** or **evidence**.

------

##### Marginal Likelihood (Evidence)

For a continuous parameter space, the marginal likelihood is given by
$$
c = f(\text{data})
= \int f(\text{data} \mid \theta)\, f(\theta)\, d\theta
= \int \mathcal{L}(\theta)\, f(\theta)\, d\theta.
$$
This quantity ensures that the posterior distribution integrates to one. While it often serves as a normalization constant in Bayesian updating, it also plays a central role in Bayesian model comparison and model selection.





### MULTIVARIATE NORMAL Distribution

For our study puprose, it is useful to introduce the multivariate normal distribution

Let $x \in \mathbb{R}^p$ be a random vector. The multivariate normal density is given by
$$
f(x)
= \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}}
  \exp\!\left(
    -\frac{1}{2}(x - \mu)^{\mathsf T}\Sigma^{-1}(x - \mu)
  \right),
$$
where $\mu \in \mathbb{R}^p$ is the mean vector and
 $\Sigma \in \mathbb{R}^{p \times p}$ is the covariance matrix.



Next step is to combine the multivariate normal distribution with the bayesian inference:

## Multivariate Normal Likelihood combine with Bayesian

Assume we observe $n$ independent $p$-dimensional vectors
$$
r_1, r_2, \dots, r_n \in \mathbb{R}^p,
$$
with
$$
r_i \mid \theta, \Sigma \sim \mathcal{N}(\theta, \Sigma),
\qquad i = 1,\dots,n,
$$
where

- $\theta \in \mathbb{R}^p$ is the unknown mean vector,
- $\Sigma \in \mathbb{R}^{p \times p}$ is the covariance matrix.

------

### Joint Density (Product Form)

The joint density of the data is
$$
f(r_1,\dots,r_n \mid \theta, \Sigma)
=
\prod_{i=1}^n
(2\pi)^{-p/2} |\Sigma|^{-1/2}
\exp\!\left\{
-\frac{1}{2}
(r_i - \theta)^{\mathsf T}\Sigma^{-1}(r_i - \theta)
\right\}.
$$

------

### Simplified Likelihood

This can be rewritten as
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




####Conditional Normal Form (Key Bayesian Result)

Ignoring terms that do not depend on $\theta$, and rewritten in matrix form, the likelihood implies
$$
f(R \mid \theta, \Sigma)
\propto
\exp\!\left\{
-\frac{n}{2}
(\theta - \bar r)^{\mathsf T}\Sigma^{-1}(\theta - \bar r)
\right\}.
$$
Thus, **as a function of $\theta$**,
$$
\theta \mid R, \Sigma
\;\propto\;
\mathcal{N}
\!\left(
\bar r,
\frac{1}{n}\Sigma
\right).
$$

### BlackLitterman Model

### 1. Start from the Return-Generating Process

Let $r_t \in \mathbb{R}^n$ denote the vector of asset returns at time $t$:
$$
r_t \mid \mu, \Sigma \sim \mathcal{N}(\mu, \Sigma),
$$
where:

- $\mu$ is the **unknown true expected return vector**,
- $\Sigma$ is the return covariance matrix.

This is the same data-generating model used earlier when we wrote
$$
r_i \mid \theta, \Sigma \sim \mathcal{N}(\theta, \Sigma).
$$
Here, simply identify:
$$
\theta \;\longleftrightarrow\; \mu
$$

------

### 2. Why We *Do Not* Use Sample Means in Black–Litterman

In classical Bayesian estimation, we would update $\mu$ using the sample mean:
$$
\bar r = \frac{1}{T}\sum_{t=1}^T r_t,
$$
leading to
$$
\mu \mid \bar r
\sim
\mathcal{N}
\!\left(
\bar r,
\frac{1}{T}\Sigma
\right).
$$
Black–Litterman replaces this *historical-data likelihood* with something more economically meaningful: **market equilibrium + investor views**.

------

### 3. The Prior: $\pi$ and $\tau\Sigma$

#### Equilibrium Expected Returns

The vector $\pi$ represents **market-implied equilibrium returns**, typically obtained via reverse optimization:
$$
\pi = \lambda \Sigma w_{\text{mkt}},
$$
where:

- $w_{\text{mkt}}$ are market-cap weights,
- $\lambda$ is the market risk-aversion coefficient.

#####3.1 Start from the Mean–Variance Optimization Problem

An investor chooses portfolio weights $w \in \mathbb{R}^n$ to maximize expected utility:
$$
\max_{w}
\;\;
w^{\mathsf T}\mu
-
\frac{\lambda}{2}
w^{\mathsf T}\Sigma w,
$$
where:

- $\mu$ = expected return vector,
- $\Sigma$ = covariance matrix of returns,
- $\lambda > 0$ = risk-aversion coefficient.

This is the standard quadratic utility / Markowitz problem.

##### 3.2 First-Order Condition

Take the derivative with respect to $w$ and set it to zero:
$$
\mu - \lambda \Sigma w = 0.
$$
Rearranging gives
$$
\mu = \lambda \Sigma w.
$$
This is the **optimality condition**:
expected returns must compensate exactly for marginal risk.

The vector $\pi$ represents **market-implied equilibrium returns** plays the role of the prior mean:
$$
\mu \sim \mathcal{N}(\pi, \tau\Sigma).
$$

#### Interpretation of $\tau\Sigma$

- $\Sigma$ captures **return co-movement**,
- $\tau$ scales **uncertainty in equilibrium returns**.

Note: $\tau$  is a parameter introduced in the Black Litterman original paper taht plays the same role as $1/T$ in the sample-mean posterior variance.

------

### 4. The Likelihood: $P$ and $Q$ as “Soft Data”

Instead of observing returns directly, BL assumes we observe **linear views** and used that to replace the value in the Bayesian formula:
$$
Q = P\mu + \varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0,\Omega).
$$
Interpretation:

- $P$: selects or combines assets (e.g., “asset 1 will outperform asset 2”),
- $Q$: the **expected return implied by the view**,
- $\Omega$: confidence in the view (smaller = stronger belief).

This replaces the classical likelihood
$$
\bar r \mid \mu \sim \mathcal{N}\!\left(\mu, \frac{1}{T}\Sigma\right)
$$
with
$$
Q \mid \mu \sim \mathcal{N}(P\mu, \Omega).
$$

------

### 5. Posterior: Same Formula, New Meaning

Because both the prior and the “data” are Gaussian, the posterior remains Gaussian:
$$
\mu \mid Q
\sim
\mathcal{N}(\mu_{\text{BL}}, \Sigma_{\text{BL}}),
$$
with
$$
\mu_{\text{BL}}
=
\pi
+
\tau\Sigma P^{\mathsf T}
\left(
P\tau\Sigma P^{\mathsf T}+\Omega
\right)^{-1}
(Q - P\pi).
$$

##Summary

**Black–Litterman replaces historical sample means with economically motivated “soft observations” of returns.**

- $\pi$: what the market implies about returns,
- $P, Q$: what the investor believes about returns,
- $\tau\Sigma, \Omega$: how confident each side is.

Mathematically, nothing new is happening —**it is the same Bayesian update people already derived for decades**, just with a better-behaved likelihood.

The Black–Litterman model can be fully understood as a direct application of Bayesian updating and Gaussian conditioning, and the references above provide both the theoretical foundation and practical insight needed to move from equations to implementation.



## References and Further Reading

1. **Ren-Raw Chen, Shih-Kuo Yeh, and Xiaohu Zhang**
    *On the Black–Litterman Model: Learning to Do Better*
2. **Stephen Satchell etc**
    *A Demystification of the Black–Litterman Model: Managing Quantitative and Traditional Portfolio Construction*
    *Journal of Asset Management*, received January 20, 2000.
3. **Brian Junker**
    *Basics of Bayesian Statistics*
    Carnegie Mellon University lecture notes.
    Available at:
    [https://www.stat.cmu.edu/~brian/463-663/week09/Chapter%2003.pdf](https://www.stat.cmu.edu/~brian/463-663/week09/Chapter 03.pdf)
4. **Sayan Mukherjee**
    *Useful Properties of the Multivariate Normal*
    Duke University lecture notes (STA 613).
    Available at:
    https://www2.stat.duke.edu/~sayan/Sta613/2018/lec/Bayesreg.pdf

------


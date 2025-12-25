### Black-Litterman Model Step by step(I)

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

The marginal likelihood is defined as:
$$
p(r) = \int p(r \mid \mu)\, p(\mu)\, d\mu
$$
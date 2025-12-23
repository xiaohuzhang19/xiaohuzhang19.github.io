### Black-Litterman Model Step by step(I)

Two years ago, inspired by the advancedment in 401K and pension fund investment, I start working on a reserch problem that well-known almost two decades old by Fisher Black and Bob Litterman at Goldman Sachcs. 

While the problem is well-known, the mystery math behind it and how to use it in pratice is barelly discussed. 

I will start the series of blog to provide a demystification of the Black–Litterman model.

## Math behind the model

we start with the Bayes theorem:
$$
p(B|A)=\frac{P(A|B)P(B)}{P(A)}
$$
The proof of bayes is relatively very straightforward.
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

 $p(B)$ is the unconditional (marginal) probability of the event of interest, also known as the prior information.
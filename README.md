# Statistical Learning assignments

During the course of statistical learning several homeworks were assigned. This repository contains both the code and the presentations of such homeworks, some sparse notes relative to the course might be present.

## Homework 1: Montecarlo Simulation

The objective of this homework was to use montecarlo simulation to verify a statistical property or behavior of our chosing. I decided to focus on the behaviour of the [Kolmogorov-Smirnov test](https://it.wikipedia.org/wiki/Test_di_Kolmogorov-Smirnov), a non parametric test used to compare a sample with a reference probability distribution. I wanted to see how the behaviour of the test changes with respect to the sample distribution and sample size.

![results](/images/KS_test_results.jpg "Results")

## Homework 2: Kernel density estimation and kernel regression

The objective of this homeowork was to understand the porpouse of the kernel methods and prepair a presentation on an interesting aspect. The presentation can focus on theory, applications, computation or simulation. My attention was focused on [kernel regression](https://en.wikipedia.org/wiki/Kernel_regression) and its naive implementation.

**The kernel regression implemented is compatible with the scikit-learn framework.** This means it can used, like any other regressor of scikit-learn, in its grid search and cross validation functions.

A Montecarlo Simulation was setuped to evaluate the difference between kernel regression and KNN. The results seem to be in favor of the kernel regression.

![results](/images/1000_700png.png )

Several other properties of kernel regression were evaluated like what type of kernel is the best, how does this regression perform when data is not sampled uniformly and also its behaviour in case of heteroscedasticity.

![results](/images/Screenshot%20from%202023-04-03%2021-07-53.png)

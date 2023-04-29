


Random intresting stuff

1. **Overfitting in machine learning**: model is too complex and fits the training data too closely, leading to poor generalization to new data.

2. **The effect of introducing too many variables in the model**: this can lead to overfitting due to the curse of dimensionality. This is an intresting way of thinking of complexity in the model. We have an encrease in complexity because we introduce some non linearity (as for polinomial regression) or becuase we increase the numbre of feature. In both cases this increases the variance of the model resuliting in poor accuracy if the number of data is not high enough.

3. **The problem of multicollinearity in multiple linear regression**: predictor variables are highly correlated, this leads to unstable and unreliable estimates of regression coefficients and their standard errors. 

4. **The effect of multicollinearity on the X^T X matrix**:  it can become close to singular or singular, leading to difficulty in estimating regression coefficients and their standard errors.

5. **Ridge regression helps with multicollinearity**: By adding a penalty term to the cost function that shrinks the coefficients towards zero, Ridge Regression reduces the variance of the estimates, making them more stable and less sensitive to small changes in the data.
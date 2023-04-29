

**Correlation matrix**: One of the most common ways to detect multicollinearity is by examining the correlation matrix between the predictor variables. If the correlation coefficient between two variables is high (close to 1 or -1), it suggests that there is a strong linear relationship between them, which can lead to multicollinearity.

**Variance Inflation Factor** (VIF): VIF measures how much the variance of an estimated regression coefficient is inflated due to multicollinearity in the model. VIF values greater than 5 or 10 suggest the presence of multicollinearity.

**Eigenvalues**: Another way to detect multicollinearity is by examining the eigenvalues of the correlation matrix. If one or more eigenvalues are close to zero, it indicates that there is a linear relationship between the variables. If there is a linear relationship between two or more predictors, the determinant of the covariance or correlation matrix is close to zero, and hence one or more of the eigenvalues is close to zero. 

**Condition number**: The condition number is the ratio of the largest eigenvalue to the smallest eigenvalue of the correlation matrix. A condition number greater than 30 suggests that there may be multicollinearity.

**Variance proportions**: Examining the proportion of variance explained by each principal component can also help detect multicollinearity. If the first few principal components explain a large proportion of the variance, it suggests that there is multicollinearity.
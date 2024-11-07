Linear regression is a supervised learning method used to estimate the relationship between a target variable and a set of features. It assumes that the target response is a linear combination of the features.

If the training set consists of $n$ observations, and each observation $i$ includes a scalar response $y_i$ and a column vector $\mathbf{x}_i$ of $p$ parameters called regressors, then the  target response $y_i$ is a linear combination of the regressors.
$$
y_i = \mathbf{x}_i \cdot \mathbf{w} + b + \epsilon_i
$$
where $\mathbf{w}$ is the parameters vectors representing the influence of each regressor on the response, $b$ is the intercept term, and $\epsilon_i$ is a scalar quantity representing influences upon the response from sources other than the explanatory variables $\mathbf{x}_i$.
$$
\mathbf{y} = \mathbf{X} \cdot \mathbf{w} + \mathbf{\epsilon}
$$
The term $b$ can be absorbed into $\mathbf{w}$ after adding a dummy variable $x_0 = 1$ in $\mathbf{x}_i$, and the relationship can be written for the entire set of $n$ observations. The $n \times p$ matrix $\mathbf{X}$ is called the design matrix, $\mathbf{y}$ is an $n \times 1$ vector of response variables, and $\mathbf{\epsilon}$ is an $n \times 1$ vector of unobserved random errors.

A linear regression model estimates the parameter vector $\mathbf{w}$ such that the resulting model has the best fit according to a loss function $Q$, usually the sum of squares of residuals along with a regularization term.
$$
\underset{\mathbf{w}}{\min} \, Q(\mathbf{w})
$$

## Ordinary least squares
Ordinary least squares is the simplest parameter estimation method for linear regression models based on minimizing the sum of squares of the residuals (a residual being the difference between the observed targets in the dataset and the targets predicted by the linear approximation).
$$
\mathbf{w} = \underset{w}{\min} \, ||\mathbf{X} \cdot \mathbf{w} - \mathbf{y}||_2^2
$$
Ordinary least squares is typically solved by a closed-form equation based on the singular value decomposition of the design matrix. This method has complexity $O(n \times p^2)$.
$$
\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$
Ordinary least squares assume that the regressors in the $\mathbf{X}$ are linearly independent. If the regressors are multicorrelated, the least squares estimate becomes highly sensitive to random errors in the observed target, resulting in a high variance model.

It is possible to constrain all estimated parameters to be non-negative, which can be useful when they represent some naturally non-negative quantities.

## Ridge regression
Ridge regression is a method for parameter estimation of linear regression models in scenarios where the variables are highly correlated. Ridge regression works by adding L2 regularization to the objective function, which imposes a penalty on the size of the coefficients.
$$
\mathbf{w} = \underset{w}{\min}||\mathbf{X} \cdot \mathbf{w} - \mathbf{y}||_2^2 + \lambda ||\mathbf{w}||_2^2
$$
When the variables are highly correlated, the model struggles to find the impact of each individual variable on the outcome. As a result, it might assign a large positive coefficient to one dependent variable and a large negative coefficient to another dependent variable.

Therefore, large coefficients are indicative of multicollinearity. These large coefficients are problematic since a small change in the corresponding variable results in a huge fluctuation in the outcome, when the impact should have been insignificant.

By imposing an L2 penalty on the size of the coefficients, the model is forced to select smaller coefficients, minimizing the impact of correlated variables. The amount of shrinkage is controlled by the $\lambda$ parameter.

## LASSO regression
LASSO is a method for parameter estimation of linear regression models in scenarios where the regressors are highly correlated. LASSO works by adding L1 regularization to the objective function, which is able to force many coefficients to zero, unlike L2 regularization.
$$
\mathbf{w} = \underset{w}{\min}||\mathbf{X} \cdot \mathbf{w} - \mathbf{y}||_2^2 + \lambda ||\mathbf{w}||_1
$$
LASSO estimation is preferable when the true model is known to be sparse. Under certain conditions, it can recover the exact set of non-zero coefficients. As the LASSO yields sparse model, it can be used to perform feature selection. The degree of sparsity is controlled by the $\lambda$ parameter.

## Elastic net regression
Elastic net is a method for parameter estimation of linear regression models that linearly combines the L1 and L2 penalties of the LASSO and ridge regression methods.
$$
\mathbf{w} = \underset{w}{\min}||\mathbf{X} \cdot \mathbf{w} - \mathbf{y}||_2^2 + \lambda \left(\rho||\mathbf{w}||_1\ + (1-\rho)||\mathbf{w}||_2^2\right)
$$
The elastic net method overcomes the limitations of the LASSO method when the variables are highly correlated. LASSO tends to select one variable randomly and ignore the others, resulting in high variance of the model. Elastic net overcomes this by adding an L2 penalty.

The L2 penalty makes the loss function strongly convex, and therefore, it has a unique minimum. The ratio of the L1 penalty to L2 penalty is controlled by the $\rho$ parameter, while the degree of regularization is controlled by the $\lambda$ parameter.

## Least angle regression
Least angle regression is a method for parameter estimation of linear regression models under high-dimensional data. It is a stepwise process that selects the regressors most highly correlated with the target variable $y$ in addition to estimating their coefficients.

1. Start with all coefficients $w$ set to zero.
2. Find the variable $x_i$ most correlated with $y$.
3. Increase the coefficient $w_i$ in the direction of the sign of its correlation with $y$. Take $r = y - \hat{y}$ along the way. Stop when some other variable $x_j$ has as much correlation with $r$ as $x_i$.
4. Increase $(w_i,w_j)$ in their joint least squares direction until some other variable $x_k$ has the same amount of correlation with the residual $r$.
5. Repeat this process until the addition of more variables does not improve the model to a statistically significant extent.

LARS is numerically efficient when the number of features is significantly greater than the number of samples. It also has the same time complexity as ordinary least squares. Moreover, it produces a full piece wise linear solution, which is useful in cross-validation.

When two features are almost equally correlated with the target, LARS increases both of them at the same rate. The algorithm thus behaves as intuition would expect and is also more stable. LARS can also be easily modified to produce efficient algorithms for other methods, like LASSO.

However, if there is any amount of noise in a high-dimensional multicollinear data, there is no reason to believe that the selected variables have a high probability of being the true explanatory variables.

This problem is not unique to LARS, as it is a general problem with variable selection approaches that seek to find the underlying deterministic components. But, because LARS is based upon an iterative refitting of the residuals, it is especially sensitive to the effects of noise.

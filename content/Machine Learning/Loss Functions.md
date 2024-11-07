
A loss function is the objective function that a machine learning model seeks to optimize during the learning process. A loss function represents the performance on a single training example, whereas, the cost function represents the average value of the loss function on the entire training set.

The choice of loss functions involves considering several factors such as the nature of the problem (classification or regression), sensitivity to outliers, and convergence properties of the loss function. When in doubt, mean squared error should be used for regression tasks and cross-entropy loss should be used for classification tasks.

## Mean squared error
The mean squared error uses the L2 loss which is a loss function used for regression tasks that calculates the squared difference between the predicted and target values. Due to squaring, it exaggerates the effect of outliers, and therefore, should be used when outliers should be penalized.
$$
L_{\text{MSE}} = (f(x) - y)^2
$$

## Mean absolute error
The mean absolute error uses the L1 loss which is a loss function used for regression tasks that calculates the absolute error between the predicted values and target values. Since it is less sensitive to outliers, it is suitable when outliers or noise should not be penalized considerably.
$$
L_{\text{MAE}} = |f(x) - y|
$$

## Huber loss
The Huber loss is a loss function used for regression tasks that utilizes the L2 norm for small errors and L1 norm for large errors. It is therefore suitable when outliers should be penalized, but large outliers should be prevented from significantly distorting the loss value.
$$
L_\delta = \begin{cases}
\frac{1}{2}(y - f(x))^2 &\text{ for } |y - f(x)| \le \delta \\
\delta|y - f(x)| - \frac{1}{2} \delta^2 &\text{ otherwise}
\end{cases}
$$

## Cross-entropy loss
The cross-entropy loss, or log loss, is a loss function used for classification tasks where the output is a set of probability values between $0$ and $1$. It measures the difference between the predicted and true probability distributions.

For binary classification with a true label $y \in \{0,1\}$ and a probability estimate $p = \Pr(y =1)$, the log loss is the negative log-likelihood of the classifier given the true label. This definition can be easily extended to multiclass classification.
$$
L_{\text{log}} = -(y \log(p) + (1-y)\log(1-p))
$$
Similar to mean square error (which is applicable for regression tasks), cross-entropy loss heavily penalizes large errors in classification tasks, while being lenient to small errors.

## Hinge loss
The hinge loss is a loss function used for classification tasks where the classification relies on a decision boundary. Hinge loss tries to maximize the margin between the decision boundary and data points, hence it is used for maximum-margin classification, notably in support vector machines.

For an intended output $t = \pm 1$ (representing the correct side of the decision boundary) and a classifier score $y$, the hinge loss is defined as the maximum of zero and $1 - t \cdot y$. This definition can be easily extended to multiclass classification.
$$
L_{\text{hinge}} = \max(0, 1 - t \cdot y)
$$
When $t$ and $y$ have the same sign and $|y| \ge 1$ (correct prediction and by enough margin), hinge loss is zero. When they have opposite sign (wrong prediction) or when they have same sign but $|y| \lt 1$ (correct prediction but not by enough margin), hinge loss increases linearly with $y$.

## Zero-one loss
The zero-one loss is a loss function used for classification tasks that simply represents the number of wrong predictions, without regard to how far off the prediction is. It is a very simple loss function that effectively represents the misclassification rate.
$$
L_{0/1} = \begin{cases}
0 &\text{ if } y = \hat{y}\\
1 &\text{ if } y \ne \hat{y}
\end{cases}
$$

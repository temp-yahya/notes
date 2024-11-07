Machine learning is a subfield of artificial intelligence which focuses on the development of algorithms that learn from data and generalize to unseen data. Therefore, it allows to perform tasks without explicit instructions.

Learning in this context is described by the widely quoted statement: "A computer is said to learn from experience $E$ with respect to some task $T$ and performance measure $P$ if its performance at the task $T$, as measured by $P$, improves with experience $E$".

## Approaches
Machine learning approaches are traditionally divided into three broad categories, depending on the nature of the feedback available to the learning system:
- Supervised learning: The algorithm learns from labelled data, using input-output pairs to find patterns and make predictions on unseen, unlabeled data.
- Unsupervised learning: The algorithm learns from unlabeled data, identifying patterns and grouping similar data points together without any predefined labels.
- Reinforcement learning: An agent takes actions in an environment to maximize some reward, learning through trial-and-error and feedback from its actions.

## No free lunch theorem
The no free lunch theorem for machine learning roughly states that no machine learning algorithm is universally better than any other algorithm, when its performance is averaged over all possible tasks. Thus, all machine learning algorithms are only as good as a randomly guessing function.

The catch however is that, NFL theorem holds only when the performance is averaged over all possible tasks. For a particular task, some algorithms vastly outperform the others. It implies that in the absence of meaningful assumptions about the task or underlying data distribution, there is no reason to prefer one learning algorithm over another.

## Inductive bias
Inductive bias is the set of assumptions that a machine learning algorithm makes about the relationship between the input features and output labels based on the training data. Inductive bias is anything that makes the algorithm learn one pattern instead of another pattern.

In machine learning, the learner is supposed to approximate the correct values for unseen data. Without any additional assumptions, this problem can not be solved, since the unseen data might have any arbitrary output value. Inductive bias encompasses these necessary assumptions.

## Occam's razor
Occam's razor is a problem-solving principle which suggests that complex hypothesis makes more assumptions, which makes them too narrow and hence do not generalize well. Therefore, when all else is equal, simpler hypothesis should be preferred.

In machine learning, it suggests that simpler models with fewer coefficients should be preferred over complex models like ensembles, since complex models will overfit the training dataset, and perform poorly on unseen data.

However, ensembles are the models of choice when prediction accuracy is important. This apparently contradicts Occam's razor, but the key is in "all else being equal".

If two models have the same generalization error on the same holdout set, only then the simpler one should be preferred. This does not apply to ensembles, since their generalization error continues to improve on the test set, even when there error on the training set falls to zero.

## IID variables
A collection of random variables is independent and identically distributed if each random variable has the same probability distribution as any other and all are mutually independent. A random sample is the same thing as a sequence of IID items.

Identically distributed means that there are no overall trends. In other words, the distribution does not fluctuate and all items in the sample are taken from the same probability distribution.

Independent means that the sample items are all independent events. In other words, they are not connected to each other in any way. Therefore, knowledge of the value of one item does not give any information about the value of any other item.

## Law of large numbers
The law of large numbers states that given a sample of independent and identically distributed values, the sample mean converges to the true mean. There are two different versions of the law of large numbers, called the weak law and the strong law of large numbers.

The weak law of large numbers states that for a sufficiently large sample, the sample mean $\overline{X}_n$ converges in probability to the expected value $\mu$. The weak law states that the sample average is likely to be near $\mu$, leaving the possibility that $|\overline{X}_n - \mu| \gt \epsilon$ happens an infinite number of times.
$$
\lim_{n \to \infty} \Pr(|\overline{X}_n - \mu| \lt \epsilon) = 1
$$
The strong law of large numbers states that for a sufficiently large sample, the sample mean $\overline{X}_n$ converges almost surely to the expected value $\mu$. While the weak law leaves the possibility that $|\overline{X}_n - \mu| \gt \epsilon$ can happen, the strong law shows that this almost surely will not happen.
$$
\Pr(\lim_{n \to \infty} \overline{X}_n = \mu) = 1
$$

## Bias-variance tradeoff
The bias-variance tradeoff is the conflict that arises when trying to simultaneously minimize two sources of error that prevents a learning algorithm from generalizing beyond its training set. It can be analyzed by decomposing the expected generalization error into a sum of three terms:
$$
\text{Expected error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible error}
$$
When the model complexity is low, it underfits the training set. This introduces a bias inherent to the model. When the model complexity is high, it overfits the training set. This causes high variance between each realization of the model on different training sets.

![[bias-variance.jpg | twitch | 400]]

The optimal model complexity is the sweet spot where the model is neither too simple to ignore important relationships in the training set, nor too complex that it memorizes the noise specific to a training set.

## Feature engineering
Feature engineering is a preprocessing step in machine learning that transforms raw data into a more effective set of inputs. It involves selecting, creating, transforming, and extracting data features.

This includes feature creation from existing data, imputing missing or invalid features, reducing data dimensionality through methods like principal components analysis, and selecting the most relevant features based on importance scores and correlation matrices.

## Supervised Learning
In supervised learning, an algorithm learns to predict an outcome $y$ given an input $x$ by first training on some labelled data called the training set. The training set consists of pairs $(\mathbf{x},y)$, where $\mathbf{x} \in X$ is the feature vector and  $y \in Y$ is its label.

If a joint probability distribution $P(x,y)$ over $X$ and $Y$ is assumed, then the training set consists of $n$ instances $\{(x_1,y_1),\dots,(x_n,y_n)\}$ drawn i.i.d. from $P(x,y)$.

A learning algorithm tries to learn a hypothesis $h: X \to Y$ which outputs an object $y \in Y$ given $x \in X$. If $L(y,h(x))$ is a loss function which measures the error in the predictions of $h$ from the true outcome $y$, then the risk of hypothesis $h$ is defined as the expectation of the loss function.
$$
R(h) = \mathbf{E}[L(y,h(x))] = \int L(y,h(x)) \, dP(x,y)
$$
The ultimate goal of the learning algorithm is to find a hypothesis $h^\ast :X \to Y$ among a hypothesis space $H$ for which the risk $R(h)$ is minimal.
$$
h* = \underset{h \in H}{\mathrm{argmin}} \, R(h)
$$

## Glivenko-Cantelli theorem
The Glivenko-Cantelli theorem states that an empirical distribution converges almost surely to the true distribution, by the strong law of large numbers, as the number of independent and identically distributed observations grow.

## Empirical risk minimization
Generally, the true risk $R(h)$ can not be computed since the distribution $P(x,y)$ is unknown to the learning algorithm. However, an estimate, called the empirical risk, can be calculated as the average of the loss function over the training set.
$$
R_{\text{emp}}(h) = \frac{1}{n} \sum^n_{i=1} L(y_i,h(x_i))
$$
The empirical risk can be a good estimate of the true risk, since by the Glivenko-Cantelli theorem, the empirical distribution almost surely converges to the true distribution as $n$ grows. The learning algorithm, therefore, chooses a hypothesis $\hat{h} \in H$ which minimizes the empirical risk.
$$
\hat{h} = \underset{h \in H}{\mathrm{argmin}} \, R_{\text{emp}}(h)
$$

## M-estimators
Machine learning considers several problems that have the goal of minimizing an objective function of the form below, where the parameter $w$ that minimizes $Q(w)$ is to be estimated. Each summand function $Q_i$ is associated with the $i$-th observation in the training set.
$$
Q(w) = \frac{1}{n} \sum^n_{i=1} Q_i(w)
$$
Such sum-minimization problems arise in empirical risk minimization, least squares, and maximum-likelihood estimation. The general class of estimators that arise as minimizers of sums are called M-estimators.

## Gradient descent
Gradient descent is an iterative algorithm for finding the local minimum of a differentiable function. The idea is to take repeated steps in the opposite direction of the function gradient at the current point (because this is the direction of the steepest descent) until it converges to a local minimum.
$$
x_{n+1} = x_n - \eta \nabla F(x_n)
$$
If the step size $\eta$ is too small, the algorithm will surely converge but would do so extremely slowly. Whereas, if the step size is too large, the algorithm would overshoot and diverge.

The iterations for the gradient descent can be rewritten for the case of M-estimators. In the context of empirical risk minimization, $\eta$ is called the learning rate, $Q_i$ is the value of the loss function at $i$-th example, and $Q(w)$ is the empirical risk. This version is also called the batch gradient descent.
$$
w' = w - \eta \nabla Q(w) = w - \frac{\eta}{n} \sum^n_{i=1} \nabla Q_i (w)
$$
Since most loss functions in machine learning are convex, the local minimum is also the global minimum. However, when the training set is large, evaluating the sums of gradients becomes very expensive, because it requires evaluating gradients of loss functions for all the examples.

## Stochastic gradient descent
Stochastic gradient descent is a very efficient modification of gradient descent that approximates the true gradient of the objective function $Q(w)$ by the gradient of the objective function $Q_i(w)$ at a single randomly selected sample.
$$
w' = w - \eta \nabla Q_i(w)
$$
To ensure that the algorithm uses each sample in the training set, the training set is shuffled after each iteration, and the algorithm sweeps over this set sequentially. Besides, most implementations include a regularization term that penalizes model complexity in the objective function.
$$
Q_i(w) = L(y_i,h(x_i)) + \lambda R(w)
$$
To improve the convergence rate, the learning rate can be made to gradually decay at each time step. The number of iterations along with a stopping criterion can be specified to place a hard limit on the execution time and provide a metric for the desired level of convergence, respectively.

## Mini-batch gradient descent
Mini-batch gradient is another modification of gradient descent that uses the gradient over a small randomly selected subset of data called a mini-batch, unlike batch gradient descent, which uses the entire dataset, and stochastic gradient descent, which uses a single sample.

Mini-batch gradient descent provides faster iterations compared to batch gradient descent, and faster convergence rate than stochastic gradient descent since it can utilize hardware optimization of matrix operations rather than computing each step separately.

## Dataset split
The training dataset is used to fit the model's parameters during the learning process. However, most approaches that search through training data for empirical relationships tend to overfit the data. To avoid this, a second holdout set called the validation dataset is used.

The validation dataset is used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters. The evaluation becomes more biased as skill on the validation dataset is incorporated into the model configuration.

The test dataset is independent of the training dataset, but follows the same probability distribution as the training dataset. It is used to provide an unbiased evaluation of the final model fit on the training dataset, and therefore, measures the generalization of the model.

## Linear classifier
A linear classifier makes the classification decision based on the value of a linear combination of the feature vectors. A linear classifier uses either the generative or discriminative approach to learn $w_i$ (parameters) which are then used with a function $f$ (often a threshold function) to make predictions.
$$
y = f(\sum_i w_i x_i)
$$
The discriminative learning of a linear classifier uses an optimization algorithm that is given a training set with desired outputs and a loss function. Thus, the learning algorithm solves an optimization problem of the form:
$$
\underset{\mathbf{w}}{\mathrm{argmin}}\, \left( \frac{1}{n} \sum_{i=1}^n L(y_i, h(\mathbf{x}_i)) + \lambda R(\mathbf{w}) \right)
$$
where $L$ is the loss function, $R$ is a regularization function that penalizes model complexity, and $\lambda$ is a non-negative hyperparameter that controls the regularization strength. This optimization problem can be solved by methods such as stochastic gradient decent.

## Maximum-margin classifier
A linear classifier that uses a hyperplane such that the distance from the hyperplane to the nearest data point on each is maximized, is known as a maximum-margin classifier and such a hyperplane is known as the maximum-margin hyperplane.

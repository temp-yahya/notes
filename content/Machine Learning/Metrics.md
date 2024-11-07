## Accuracy paradox
The accuracy paradox states that accuracy is not a good metric for classifier models when dealing with imbalanced data. For example, if category A is dominant, being found in 99% of all cases, then a no-skill model that simply predicts that all cases are category A will have an accuracy of 99%.

## Precision and recall
Precision and recall are performance metrics used for classification models when dealing with imbalanced data. Precision is also known as positive predictive value, and recall is also known as sensitivity in binary classification.

Precision measures the accuracy of the positive predictions, indicating the proportion of correctly predicted positive cases out of all instances predicted as positive. Precision, therefore, emphasizes the quality of the model's predictions. For binary classification, precision is defined as:
$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$
Recall measures the ability of a model to capture all positive instances, indicating the proportion of correctly predicted positive cases out of all actual positive cases. Recall, therefore, focuses on the completeness of the model's predictions. For binary classification, recall is defined as:
$$
\text{Recall} = \frac{TP}{TP + FN}
$$
The precision-recall trade-off represents the inverse relationship between precision and recall when adjusting the decision threshold of a classifier. Maximizing precision will minimize the number of false positives, whereas maximizing recall will minimize the number of false negatives.

## F-score
The F-score is a performance metric that evaluates the balance between precision and recall. It has two variants, namely the balanced $F_1$ score and the more general $F_\beta$ score. For both variants, the best value is $1.0$, and the worst value is $0.0$.

The $F_1$ score is defined as the harmonic mean of precision and recall. It thus gives equal importance to both precision and recall.
$$
F_1 = \frac{2}{\text{precision}^{-1} + \text{recall}^{-1}} = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$
The $F_\beta$ score uses a factor $\beta$ to assign different importance to precision and recall. If $\beta \gt 1$, it gives more emphasis to recall, and if $\beta \lt 1$, it gives more emphasis to precision. It is defined as the weighted harmonic mean of precision and recall.
$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{(\beta^2 \cdot \text{precision}) + \text{recall}}
$$


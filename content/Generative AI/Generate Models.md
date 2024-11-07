## Generative and discriminative models
Generative and discriminative models are two approaches to machine learning. Generative models can generate new data instances, while discriminative models simply discriminate between different categories of data instances.

A generative model learns how the data was generated in order to classify a signal. It then judges which category is most likely to generate the given signal. A discriminative model does not care about how the data was generated, and instead simply learns how to classify a given signal.

Therefore, a generative model learns the joint probability distribution $\mathrm{P}(X,Y)$, which it later transforms into $\mathrm{P}(Y|X)$ to classify the data. Whereas, a discriminative model learns the conditional probability $\mathrm{P}(Y|X)$ directly from the data.

Generative models include naive Bayes methods, hidden Markov models, linear discriminant analysis, generative adversarial networks, and diffusion models. While, discriminative models include logistic regression, support vector machines, decision trees, and random forests.

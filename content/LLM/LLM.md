Large language models are trained on massive text corpora with billions of parameters. LLMs undergo a two-stage process: initial pre-training (where they learn the fundamental language structures) followed by alignment with human values. They can then be fine-tuned on a smaller, task-specific dataset.

## Fine-Tuning
Fine-tuning uses a pre-trained model as a foundational model. The process involves further training on a smaller, domain-specific dataset.

![[llm-mindmap.png]]

There are three types of fine-tuning:
- **Unsupervised fine-tuning**: The LLM is exposed to a large corpus of unlabeled text from the target domain. This approach is useful for new domains like legal or medicine, but is less precise for specific tasks such as classification or summarization.
- **Supervised fine-tuning**: The LLM is trained on labeled data tailored to the target task. For example, fine-tuning an LLM for text classification in a business context uses a dataset of text snippets with class labels. However, obtaining the labeled data is difficult.
- **Instruction fine-tuning**: This method relies on providing the LLM with natural language instructions, useful for creating specialized assistants. It reduces the need for vast amounts of labeled data but depends heavily on the quality of the prompts.

| Aspect             | Pre-training                                                         | Fine-tuning                                                                                                     |
| ------------------ | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Definition         | Training on a large corpus of unlabeled text                         | Adapting a pre-trained model to specific tasks                                                                  |
| Data requirement   | Extensive and diverse unlabeled data                                 | Smaller and task-specific labeled data                                                                          |
| Objective          | Build general linguistic knowledge                                   | Specialize model for specific tasks                                                                             |
| Process            | Data collection, training on large dataset, and predicting next word | Task-specific data collection, modify last layer for task, train on new dataset, generate output based on tasks |
| Model modification | Entire model trained                                                 | Last layers adapted for new tasks                                                                               |
| Computational cost | High (large dataset, complex model)                                  | Lower (smaller dataset, fine-tuning layers)                                                                     |
| Training duration  | Weeks to months                                                      | Days to weeks                                                                                                   |
| Purpose            | General language understanding                                       | Task-specific performance improvement                                                                           |
| Examples           | GPT, LLaMA 3                                                         | Fine-tuning LLaMA 3 for summarization                                                                           |

## Retrieval Augmented Generation
A popular method to utilize your own data is by incorporating it into the prompt when querying the LLM model. This approach, known as RAG, involves retrieving relevant data and using it as additional context for the LLM.

Instead of depending solely on knowledge from the training data, a RAG architecture pulls relevant information from a small amount of organization's own data. This process avoids the cost associated with fine-tuning or pre-training the model.

![[rag-pipeline.png]]

A traditional RAG pipeline consists of these steps:
1. **Data indexing**: The data is organized efficiently for quick retrieval. This involves processing, chunking, and storing data in a vector database using indexing strategies like search indexing, vector indexing, and hybrid indexing.
2. **Input query processing**: Refine user queries to improve compatibility with indexed data. This can include simplification or vector transformation of queries for enhanced search efficiency.
3. **Searching and ranking**: Retrieve and rank data based on relevance using search algorithms such as TF-IDF, BM25, and deep learning models like BERT to interpret the query's intent and context.
4. **Prompt augmentation**: Incorporate relevant information from the search results into the original query to provide the LLM with additional context, enhancing response accuracy and relevance.
5. **Response generation**: Use the augmented prompt to generate responses that combine the LLM's knowledge with current, specific data, ensuring high-quality, contextually grounded answers.

## Choosing Between RAG and Fine-Tuning
When considering external data access, RAG is likely a superior option for applications needing to access data sources. Fine-tuning, on the other hand, is more suitable if you require the model to adjust its behavior, writing style, or incorporate domain-specific knowledge.

In terms of suppressing hallucinations and ensuring accuracy, RAG systems tend to perform better as they are less prone to generating incorrect information. However, if you have ample domain-specific labeled training data, fine-tuning can result in a more tailored model behavior, whereas RAG systems are robust alternatives when such data is scarce.

RAG systems provide an advantage with dynamic data retrieval capabilities for environments where data frequently changes. Additionally, it is crucial to ensure the transparency of the model's decision-making process.

![[rag-vs-fine-tuning.png]]

## Fine-Tuning Pipeline
Fine-tuning an LLM involves seven stages, each essential for adapting the pre-trained model to specific tasks and ensuring optimal performance.

![[fine-tuning-pipeline.png]]

### 1. Dataset Preparation
Fine-tuning starts with adapting the pre-trained model for specific tasks by updating its parameters using a new dataset. This involves cleaning and formatting the dataset to match the target task. The dataset is composed of user input-output pairs. The input is the user's query while the output is the expected model's response.

### 2. Model Initialization
Model initialization is the process of setting up the initial parameters and configurations of the LLM before training or deploying it. This step is crucial for ensuring that the model performs optimally, trains efficiently, and avoids issues such as vanishing or exploding gradients.

### 3. Training Setup
Setting up the training environment for LLM fine-tuning involves configuring the necessary infrastructure to adapt a pre-existing model for specific tasks. This includes selecting relevant training data, defining the model's architecture and hyperparameters, and running training iterations to adjust the model's parameters.

### 4. Fine-Tuning Techniques
Fine-tuning updates all parameters of the model, ensuring comprehensive adaption to the new task. Alternatively, half fine-tuning (HFT) or parameter-efficient fine-tuning (PEFT) approaches, such as using adapter layers, can be employed to partially fine-tune the model. This method attaches additional layers to the pre-trained model, allowing for efficient fine-tuning with fewer parameters.

### 5. Evaluation and Validation
Evaluation and validation involve assessing the fine-tuned LLM's performance on unseen data to ensure it generalizes well and meets the desired objectives. Evaluation metrics, such as cross-entropy, measure prediction errors, while validation monitors loss curves and other performance indicators to detect issues like overfitting.

### 6. Deployment
Deploying an LLM means making it operational and accessible for specific applications. This involves configuring the model to run efficiently on designated hardware or software platforms. Deploying also includes setting up integrations, security measures, and monitoring systems to ensure reliable and secure performance.

### 7. Monitoring and Maintenance
Monitoring and maintaining an LLM after deployment is crucial to ensure ongoing performance and reliability. This involves continuously tracking the model's performance, addressing any issues that arise, and updating the model as needed to adopt to new data or changing requirements.

## Data Preparation
These are the steps involves in data preparation:
1. **Data collection**: The first step is to collect data from various sources. These sources can be in any format, such as web pages, CSV files, or SQL databases. Python provides several libraries for various data formats:
	- CSV files: `pandas`
	- Web pages: `beautifulsoup4` and `requests`
	- SQL databases: `sqlalchemy`
	- S3 storage: `boto3`
	- Data integration: `rapidminer`
	- Data cleaning: `trifacta`
2. **Data preprocessing and formatting**: This step involves tasks such as cleaning the data, handling missing values, and formatting the data to the match the specific task. Several libraries can be used for data preprocessing, such as `spacy`, `nltk`, `transformers`, and `knime`.
3. **Handling data imbalance**: This step is crucial for ensuring balanced performance across all classes. Several techniques can be used:
	- **Over-sampling and under-sampling**: Techniques like SMOTE generate synthetic examples to achieve balance. Python library: `imbalanced-learn`.
	- **Adjusting loss function**: Modify the loss function to give more weight to the minority class, setting class weights inversely proportional to the class frequencies.
	- **Focal loss**: A variant of cross-entropy loss that adds a factor to down-weight easy examples and focus training on hard negatives. Python library: `focal_loss`.
	- **Cost-sensitive learning**: Incorporating the cost of misclassifications directly into the learning algorithm, assigning a higher cost to misclassifying minority samples.
	- **Ensemble methods**: Using techniques like bagging and boosting to combine multiple models and handle class imbalance. Python library: `sklearn.ensemble`.
4. **Splitting dataset**: This step involves splitting the dataset into training and validation sets, typically using an 80:20 ratio. Different techniques are:
	- **Random sampling**: Selecting a subset of data randomly to create a representative sample. Python library: `sklearn.model_selection.train_test_split`.
	- **Stratified sampling**: Dividing the dataset into subgroups and sampling from each to maintain class balance. Python library: `sklearn.model_selection.StratifiedShuffleSplit`.
	- **K-fold cross-validation**: Splitting the dataset into K folds and performing training and validation K times. Python library: `sklearn.model_selection.KFold`.
	- **Leave-one-out cross-validation**: Using a single data point as the validation set and the rest for training, repeated for each data point. Python library: `sklearn.model_selection.LeaveOneOut`.

### Data Annotation
Data annotation involves labeling or tagging textual data with specific attributes relevant to the model's training objective. This process is crucial for supervised learning tasks and greatly influences the performance of the fine-tuned model.

There are several approaches to data annotations:
- **Human annotation**: Manual annotation by human experts remains a gold standard due to its accuracy and context understanding. However, it is time-consuming and costly for large datasets. Tools like Microsoft Excel, Prodigy, and Innodata facilitate this process.
- **Semi-automatic annotation**: Combining machine learning algorithms with human review to create labeled datasets more efficiently. Tools like Snorkel use weak supervision to generate initial labels, which are then refined by human annotators.
- **Automatic annotation**: Fully automated annotation leverages machine learning algorithms to label data without human intervention. Services like Amazon SageMaker Ground Truth utilize machine learning to automate data labeling, although the accuracy may vary depending on the complexity of the task.

### Data Augmentation
Data augmentation techniques expand training datasets artificially to address data scarcity and improve model performance. Advanced techniques include:
- **Word embeddings**: Use word embeddings like Word2Vec and GloVe to replace words with their semantic equivalents, thereby generating new data instances.
- **Back translation**: Translating text to another language and then back to the original language to create paraphrased data. This technique helps in generating diverse training samples. Tools like Google Translate API are commonly used for this purpose.
- **Adversarial attacks**: Generating augmented data through adversarial examples that slightly modify the original text to create new training samples while preserving the original meaning. Libraries like TextAttack provide frameworks for such augmentations.
- **NLP-AUG**: This library offers a variety of augmenters for character, word, sentence, audio, and spectrogram augmentation, enhancing data diversity.

### Synthetic Data Generation Using LLMs
LLMs can generate synthetic data through innovative techniques such as:
- **Prompt engineering**: Crafting specific prompts to guide LLMs like GPT-3 in generating relevant and high-quality synthetic data.
- **Multi-step generation**: Employing iterative generation processes where LLMs generate initial data that is refined through subsequent steps. This method can produce high-quality synthetic data for various tasks, including summarization and bias detection.

### Challenges in Data Preparation
Key challenges in data preparation include:
- **Domain relevance**: Ensuring that the data is relevant to the specific domain for specific model performance. Mismatched domain data can lead to poor generalization and inaccurate outputs.
- **Data diversity**: Including diverse and well-balanced data to prevent model biases and improve generalization. A lack of diversity can cause the model to perform poorly on under-representated scenarios.
- **Data size**: Collecting a large dataset, with at least 1000 samples, is recommended for effective fine-tuning.
- **Data cleaning and preprocessing**: Removing noise, errors, and inconsistencies are critical for providing clean inputs to the model. Poorly preprocessed data can degrade model performance significantly.
- **Data annotation**: Ensuring precise and consistent labeling is essential for tasks requiring labeled data. Inconsistent annotation can lead to unreliable model predictions.
- **Handling rare cases**: Adequately representing rare but important instances in the dataset to ensure the model can generalize to less frequent but critical scenarios.
- **Ethical considerations**: Scrutinizing data for harmful or biased content to prevent unintended consequences.

### Best Practices
There are some best practices in data preparation:
- **High quality data collection**: Ensuring high quality, diverse, and representative data is critical. Tools like DataRobot Paxata and KNIME Analytics Platform offer robust data profiling and transformation capabilities.
- **Effective data preprocessing**: Proper data preprocessing is essential for model performance. Utilizing libraries like `spacy`, `nltk`, and `transformers` can streamline preprocessing tasks. Platforms like Trifacta Wrangler and RapidMiner automate data cleaning tasks, improving efficiency and ensuring consistency.
- **Managing data imbalance**: Addressing data imbalance is crucial. Techniques like over-sampling, under-sampling, and SMOTE help balance datasets. Libraries like `imbalanced-learn` and ensemble methods in `scikit-learn` provide robust tools for managing imbalanced datasets.
- **Augmenting and annotating data**: Data augmentation and annotation improves model robustness. Tools like NLP-AUG, TextAttack, and Snorkel offer sophisticated capabilities for creating diverse and well-balanced datasets.
- **Ethical data handling**: Ensuring ethical data handling involves thorough scrutiny for biases and privacy concerns. Implementing privacy-preserving techniques and filtering harmful content is critical. Services like Amazon SageMaker Ground Truth ensure scalable and secure data annotation.
- **Regular evaluation and iteration**: Continuous evaluation and iteration of the data preparation pipeline help maintain data quality and relevance. Leveraging feedback loops and performance metrics ensures ongoing improvements and adaptation to new data requirements.

## Model Initialization
These are the steps involved in initializing a large language model:
1. **Set up the environment**: Configure your environment, such as setting up GPU/TPU usage if available, which can significantly speed up model loading and inference.
2. **Install the dependencies**: Ensure that all necessary software and libraries are installed. This typically includes package managers like pip and frameworks like PyTorch or Tensorflow.
3. **Import the libraries**: Import the required libraries in your code. Common libraries include HuggingFace transformers, PyTorch, and other utility libraries.
4. **Choose the language model**: Select the appropriate pre-trained language model based on your task requirements. This could be models like BERT, GPT-3, or others available on platforms like HuggingFace Model Hub.
5. **Download the model**: Use the chosen framework's functions to download the pre-trained model from an online repository.
6. **Load the model**: Load the model into memory, ready for inference or further fine-tuning. This step ensures the model weights are initialized and ready for use.
7. **Execute tasks**: Perform the desired tasks using the loaded model. This could involve making predictions, generating text, or fine-tuning the model on a new dataset.

### Challenges in Model Initialization
These are the challenges in model initialization:
- **Alignment with the target task**: It is essential that the pre-trained model closely aligns with your specific task or domain. This initial alignment serves as a solid foundation for further fine-tuning efforts, leading to improved efficiency and results.
- **Understanding the pre-trained model**: Before making a selection, it is crucial to thoroughly comprehend the architecture, capabilities, limitations, and the tasks the model was originally trained on. Without this understanding, fine-tuning efforts may not yield the desired outcomes.
- **Availability and compatibility**: Careful consideration of a model's documentation, license, maintenance, and update frequency is necessary to avoid potential issues and ensure smooth integration into your application.
- **Model architecture**: Not all models excel at every task. Each model architecture has its strengths and weaknesses, so selecting one aligned with your specific task is essential for favorable outcomes.
- **Resource constraints**: Loading pre-trained LLMs is resource-heavy and requires more computation. These models need high-performance CPUs and GPUs and a significant amount of disk space. For example, the Llama 3 8B model requires a minimum of 16GB memory to load and run the inference.
- **Privacy**: Privacy and confidentiality are crucial factors when selecting a large language model. Many businesses prefer not to share their data with external LLM providers. In such instances, hosting an LLM on local servers or using pre-trained LLMs available through private cloud providers can be a viable solution. These approaches ensure that data remains within the company's premises.
- **Cost and maintenance**: Hosting LLMs on local servers or utilizing cloud vendors entails significant time and expense.
- **Model size and quantization**: Utilizing a pre-trained model with high memory consumption can still be viable by employing its quantized version. Through quantization, pre-trained weights can be loaded with reduced precision, typically 4-bit, substantially reducing model size in exchange for a small reduction in accuracy.
- **Pre-training datasets**: Examine the datasets used for pre-training to gauge the model's understanding of language. These are important as there are models available specifically for performing code generation, and we do not want to use these models for finance text classification.
- **Bias awareness**: Be vigilant regarding potential biases in pre-trained models, especially if unbiased predictions are required. The bias awareness can be evaluated by testing different models and backtracking the datasets used for pre-training.

## Training Setup
These are the steps involved in training setup:
- **Setting up the training environment**: When setting up the environment for training an LLM, it is crucial to configure high-performance hardware, such as GPUs or TPUs, and ensure proper installation of necessary software components like CUDA, cuDNN, and deep learning frameworks such as PyTorch or Tensorflow, or HuggingFace Transformers.
- **Defining the hyperparameters**: When defining hyperparameters for fine-tuning an LLM, it is essential to carefully tune key parameters such as learning rate, batch size, and epochs to optimize the model's performance.
- **Initializing optimizers and loss functions**: When initializing optimizers and loss functions for fine-tuning an LLM, it is crucial to select the appropriate optimizer to efficiently update the model's weights and the correct loss function to measure model performance.

### Defining Hyperparameters
Key hyperparameters like learning rate, batch size, and epochs are crucial for enhancing the model's performance and aligning it to your particular use case:
- **Learning rate**: The learning rate dictates the speed at which the model adapts to the problem. Smaller learning rates necessitate more training due to the minimal weight adjustments per update, while larger learning rates lead to quicker changes to weights.
- **Batch size**: The batch size determines the number of samples processed before the model parameters are updated.
- **Epochs**: An epoch refers to a full pass through the entire training dataset. The number of epochs defines how many such passes need to be completed.

### Hyperparameter Tuning
LLM hyperparameter tuning involves adjusting various hyperparameters during the training process to identify the optimal combination that yields the best output. This process often entails significant trial and error, tracking each hyperparameter adjustment, and recording the resulting performance.

Conducting this manually can be highly time-consuming. To address this, automated hyperparameter tuning methods have been developed:
- **Random search**: This method randomly selects and evaluates combinations of hyperparameters from a specified range. It is a straightforward and efficient approach capable of exploring a large parameter space. However, it may not always find the optimal combination of hyperparameters.
- **Grid search**: Unlike random search, grid search exhaustively evaluates every possible combination of hyperparameters from a given range. It is resource-intensive but finds the optimal set of hyperparameters.
- **Bayesian optimization**: This method uses a probabilistic model to predict the performance of different hyperparameters and selects the best ones accordingly. It is less resource-intensive than grid search, but is less reliable in finding the optimal set of hyperparameters.

### Optimizers and Loss Functions
Choosing the right optimizer and loss function is crucial for training and fine-tuning an LLM. Below are descriptions of some commonly used optimization algorithms, their advantages, disadvantages, and appropriate use cases.

**Gradient descent**: Gradient descent is a fundamental optimization algorithm used to minimize a cost function in machine learning models. It aims to find the optimal parameters for a neural network.

**How it works**: Gradient descent iteratively updates model parameters in the direction of the negative gradient of the cost function. It calculates gradients for each parameter and applies updates across all data points until convergence. This method utilizes the entire dataset to calculate gradients, often requiring a fixed learning rate and being sensitive to the scale of data and learning rate choice.

Its advantages are:
- Simple and easy to implement
- Intuitive and easy to understand
- Converges to the global minimum for convex functions
- Suitable for small-scale problems

Its disadvantages are:
- Computationally expensive on larger datasets
- May get stuck in local minima
- Requires a large number of iterations
- Sensitive to the choice of learning rate

**When to use**: Gradient descent is best suited for small datasets where gradient computation is cheap and simplicity and clarity is preferred.

**Stochastic gradient descent**: Stochastic gradient descent is a variant of gradient descent that focuses on reducing computation per iteration.

**How it works**: SGD updates parameters using a single or few data points at each iteration, introducing randomness in updates. It reduces the computational burden per iteration and often converges faster than gradient descent. However, it requires a smaller learning rate due to high variance and benefits from momentum to stabilize updates.

Its advantages are:
- Fast and handles large datasets well
- Efficient memory usage
- Simple and easy to implement
- Can escape local minima due to noise

Its disadvantages are:
- High variance in updates can lead to instability
- Can overshoot the global minimum
- Sensitive to the choice of learning rate
- Can be slower to converge compared to batch methods

**When to use**: SGD is ideal for large datasets, incrementally learning scenarios, and real-time learning environments where computational resources are limited.

**Mini-batch gradient descent**: Mini-batch gradient descent offers a compromise between the efficiency of SGD and the stability of batch gradient descent.

**How it works**: It splits data into small batches and updates the parameters using gradients averaged over each mini-batch. This reduces variance compared to SGD and is more efficient than batch gradient descent.

Its advantages are:
- Balance between efficiency and stability
- More generalizable updates
- Reduces the variance of parameter updates
- Provides a compromise between SGD and batch GD

Its disadvantages are:
- Requires tuning of batch size
- Can still be computationally expensive for large datasets
- More complex implementation
- Can require more iterations than full-batch gradient descent

**When to use**: Mini-batch gradient descent is suitable for most deep learning tasks, especially when working with moderate to large datasets.

**AdaGrad**: Adaptive gradient algorithm is designed for sparse data and high-dimensional models, adjusting learning rates to improve performance on sparse data.

**How it works**: AdaGrad adapts the learning rate for each parameter based on historical gradient information, accumulating squared gradients. This approach prevents large updates for frequent parameters, and helps in dealing with sparse features.

Its advantages are:
- Adapts the learning rate for each parameters
- Good for sparse data
- No need to manually tune learning rates
- Works well with high-dimensional data

Its disadvantages are:
- Learning rate can diminish to zero, stopping learning
- May require more tuning for convergence
- Accumulation of squared gradients can lead to overly small learning rates
- Can slow down significantly

**When to use**: AdaGrad is useful for sparse datasets like text and images where learning rates need to adapt to feature frequency.

**RMSprop**: Root mean square propagation is an adaptive learning rate method designed to perform better on non-stationary and online problems.

**How it works**: RMSprop modifies AdaGrad by using a moving average of squared gradients to adapt learning rates based on recent gradient magnitudes. It maintains a running average of squared gradients to help in maintaining steady learning rates.

Its advantages are:
- Addresses the diminishing learning rate of AdaGrad
- Adapts learning rate based on recent gradients
- Effective for recurrent neural networks
- More robust against non-stationary targets

Its disadvantages are:
- Can still get stuck in local minima on non-convex problems
- Requires hyperparameter tuning
- Requires careful tuning of the decay rate
- Can be sensitive to the initial learning rate

**When to use**: RMSprop is best for non-convex optimization problems, training RNNs and LSTMs, and dealing with noisy or non-stationary objectives.

**AdaDelta**: 

# Model engineering


## Overview

- **Classical supervised ML** — predict known outcomes from labeled data.
- **Unsupervised learning** — discover structure where no labels exist.
- **Dimensionality reduction** — make data smaller without losing too much meaning.
- **Anomaly detection** — find what looks abnormal.
- **Recommender systems** — suggest the next best item or action.
- **Deep learning** — learn complex patterns with neural networks.
- **Foundation / LLM systems** — general-purpose pretrained models for language and multimodal tasks.
- **Reinforcement learning** — learn by trial, feedback, and reward.
- **Decision layers** — turn model scores into usable actions.



├── 1) Classical supervised ML - Models that learn from labeled examples to predict a known outcome such as a class or numeric value.
│   ├── Linear models
│   │   ├── Linear Regression
│   │   ├── Logistic Regression
│   │   ├── Ridge
│   │   ├── Lasso
│   │   ├── Elastic Net
│   │   ├── SGD Regressor / SGD Classifier
│   │   └── GLMs
│   │       ├── Poisson Regression
│   │       ├── Gamma Regression
│   │       └── Tweedie / related families
│   ├── Tree-based models
│   │   ├── Decision Tree
│   │   ├── Random Forest
│   │   ├── Extra Trees
│   │   └── Bagging models
│   ├── Boosting models
│   │   ├── Gradient Boosting
│   │   ├── AdaBoost
│   │   ├── XGBoost
│   │   ├── LightGBM
│   │   └── CatBoost
│   ├── Margin / kernel methods
│   │   ├── Linear SVM
│   │   ├── Kernel SVM
│   │   └── SVR
│   ├── Distance / instance-based
│   │   ├── k-NN classifier
│   │   └── k-NN regressor
│   ├── Probabilistic models
│   │   ├── Naive Bayes
│   │   │   ├── Gaussian NB
│   │   │   ├── Multinomial NB
│   │   │   └── Bernoulli NB
│   │   ├── Bayesian linear models
│   │   └── Probabilistic graphical models
│   └── Time-series forecasting models
│       ├── ARIMA / SARIMA
│       ├── ETS / exponential smoothing
│       ├── Prophet
│       └── State-space / Kalman-style models
│
├── 2) Unsupervised learning - Models that find hidden structure, segments, or patterns in data without needing labeled answers.
│   ├── Clustering
│   │   ├── K-Means
│   │   ├── MiniBatch K-Means
│   │   ├── DBSCAN
│   │   ├── HDBSCAN
│   │   ├── Hierarchical / Agglomerative
│   │   ├── Gaussian Mixture Models
│   │   ├── Spectral Clustering
│   │   └── Mean Shift
│   ├── Association / pattern discovery
│   │   ├── Apriori
│   │   └── FP-Growth
│   └── Topic / latent structure
│       ├── LDA topic modeling
│       ├── NMF
│       └── PLS / latent-factor methods
│
├── 3) Dimensionality reduction & feature compression - Methods that reduce the number of input features while trying to keep the most useful signal.
│   ├── Linear reduction
│   │   ├── PCA
│   │   ├── Truncated SVD
│   │   ├── ICA
│   │   └── Factor Analysis
│   ├── Nonlinear / manifold methods
│   │   ├── UMAP
│   │   ├── t-SNE
│   │   ├── Isomap
│   │   └── LLE
│   └── Learned compression
│       ├── Autoencoders
│       └── Variational Autoencoders
│
├── 4) Anomaly / outlier detection - Models that identify unusual, rare, or suspicious data points that differ from normal patterns.
│   ├── Isolation Forest
│   ├── One-Class SVM
│   ├── Local Outlier Factor
│   ├── Robust covariance / Elliptic Envelope
│   └── Reconstruction-based anomaly models
│       └── Autoencoder anomaly detection
│
├── 5) Recommender systems - Systems that predict what a user is likely to prefer, click, watch, buy, or rank highly.
│   ├── Collaborative filtering
│   │   ├── User-user
│   │   ├── Item-item
│   │   └── Matrix factorization
│   ├── Content-based recommenders
│   ├── Hybrid recommenders
│   └── Ranking / learning-to-rank
│       ├── LambdaMART
│       ├── Pairwise ranking models
│       └── Neural ranking models
│
├── 6) Deep learning - Neural-network-based models that learn complex patterns from large, high-dimensional data such as images, text, audio, or tabular data.
│   ├── Feedforward / tabular nets
│   │   ├── MLP
│   │   ├── Wide & Deep
│   │   ├── DeepFM
│   │   └── TabNet / FT-Transformer-style tabular nets
│   ├── Computer vision
│   │   ├── CNNs
│   │   │   ├── LeNet / AlexNet / VGG
│   │   │   ├── ResNet / DenseNet / EfficientNet
│   │   │   └── U-Net / segmentation nets
│   │   ├── Vision Transformers
│   │   └── Detection / segmentation models
│   │       ├── YOLO family
│   │       ├── Faster R-CNN
│   │       └── Mask R-CNN
│   ├── Sequence / time-series / speech
│   │   ├── RNN
│   │   ├── LSTM
│   │   ├── GRU
│   │   ├── Temporal CNNs
│   │   └── Transformer sequence models
│   ├── Generative models
│   │   ├── GANs
│   │   ├── VAEs
│   │   ├── Diffusion models
│   │   └── Autoregressive neural models
│   └── Graph learning
│       ├── GCN
│       ├── GraphSAGE
│       ├── GAT
│       └── Graph transformers
│
├── 7) Foundation models / LLM-based systems - Large pre-trained models that can perform broad language or multimodal tasks and power prompting, RAG, assistants, and agents.
│   ├── Text embedding models
│   │   ├── Sentence-transformer style models
│   │   ├── OpenAI embedding models
│   │   ├── Gemini Embedding 2 Preview
│   │   └── Other retrieval / reranker models
│   ├── Text-only or primarily language foundation models
│   │   ├── OpenAI GPT family
│   │   │   ├── GPT-5.4
│   │   │   ├── GPT-5.4-mini
│   │   │   ├── GPT-5.4-nano
│   │   │   ├── GPT-5.3-Codex
│   │   │   ├── GPT-5.2
│   │   │   └── earlier GPT-5 variants / predecessors
│   │   ├── Anthropic Claude family
│   │   │   ├── Claude Opus line
│   │   │   │   ├── Opus 4.6
│   │   │   │   ├── Opus 4.5
│   │   │   │   └── Opus 4.1
│   │   │   ├── Claude Sonnet line
│   │   │   │   ├── Sonnet 4.6
│   │   │   │   ├── Sonnet 4.5
│   │   │   │   └── Sonnet 3.7
│   │   │   └── Claude Haiku line
│   │   │       └── Haiku 4.5
│   │   ├── Meta Llama family
│   │   │   ├── Llama 4 Scout
│   │   │   ├── Llama 4 Maverick
│   │   │   ├── Llama 3
│   │   │   ├── Llama 3.1
│   │   │   └── related instruction / open-weight variants
│   │   ├── Google Gemini family
│   │   │   ├── Gemini 3
│   │   │   ├── Gemini general multimodal models
│   │   │   ├── Gemini Deep Research Preview
│   │   │   └── task-specific Gemini variants
│   │   ├── Other major open / commercial LLM families
│   │   │   ├── Mistral / Mixtral
│   │   │   ├── Cohere Command
│   │   │   ├── Qwen
│   │   │   ├── DeepSeek
│   │   │   ├── AI21 Jamba / Jurassic lineage
│   │   │   └── open-source research models
│   ├── Multimodal foundation models
│   │   ├── image + text
│   │   ├── audio + text
│   │   ├── video + text
│   │   └── document / PDF-aware models
│   ├── Specialized foundation-model use patterns
│   │   ├── Prompted LLM apps
│   │   ├── Fine-tuned LLM apps
│   │   ├── RAG systems
│   │   ├── Tool-using assistants
│   │   └── Agent systems
│   └── Smaller on-device / edge language models
│       ├── distilled LLMs
│       ├── quantized local models
│       └── mobile / edge deployment variants
│
├── 8) Reinforcement learning / decision-making - Models that learn actions or strategies by interacting with an environment and improving based on rewards.
│   ├── Bandits
│   │   ├── epsilon-greedy
│   │   ├── UCB
│   │   └── Thompson sampling
│   ├── Classical RL
│   │   ├── Q-learning
│   │   ├── SARSA
│   │   └── DQN
│   └── Policy-gradient / actor-critic
│       ├── REINFORCE
│       ├── A2C / A3C
│       ├── PPO
│       └── SAC / TD3
│
└── 9) Decision layers on top of models - Rules, thresholds, escalation logic, and control policies that convert raw model outputs into real business decisions.
    ├── Thresholded classifiers
    ├── Calibrated probability models
    ├── Abstain / reject-option systems
    ├── Human-in-the-loop escalation
    ├── Rules + model hybrids
    └── Multi-stage cascades




    ## Comparison table

    | Model family                                 | Typical examples                                                       | Best used for                                                             | Strengths                                           | Tradeoffs / cautions                                               |
    | -------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------ |
    | **Linear models**                            | Linear Regression, Logistic Regression, Ridge, Lasso, Elastic Net      | Simple prediction problems, strong baselines, interpretable tabular tasks | Fast, easy to explain, easy to deploy               | May underfit complex patterns                                      |
    | **Tree-based models**                        | Decision Tree, Random Forest                                           | Tabular classification/regression, mixed feature types                    | Strong performance, handles nonlinearity well       | Can become less interpretable as complexity grows                  |
    | **Boosting models**                          | Gradient Boosting, XGBoost, LightGBM, CatBoost                         | High-performing tabular tasks                                             | Often excellent accuracy on structured data         | More tuning, more complexity, harder to explain than simple models |
    | **SVMs**                                     | Linear SVM, Kernel SVM                                                 | Medium-sized classification tasks, margin-based separation                | Can work well on some clean datasets                | Harder to scale and explain in many real settings                  |
    | **Nearest-neighbor / distance-based models** | k-NN                                                                   | Small datasets, similarity-based decisions                                | Simple intuition                                    | Can be slow at inference, sensitive to scaling                     |
    | **Probabilistic models**                     | Naive Bayes, Gaussian models                                           | Text classification, simple probabilistic baselines                       | Fast, lightweight, good baseline tools              | Strong assumptions may limit performance                           |
    | **Feature-reduced models**                   | Selected-feature models                                                | When you want simpler, more maintainable models                           | Better interpretability, lower memory/training cost | Risk of dropping useful signal                                     |
    | **Dimensionality reduction models**          | PCA, Truncated SVD, ICA                                                | Compressing overlapping numeric features, preprocessing                   | Can reduce complexity and redundancy                | Reduced interpretability; not always worth it                      |
    | **Clustering models**                        | K-Means, DBSCAN, Hierarchical Clustering, GMM                          | Segmentation, pattern discovery, unlabeled data                           | Useful for exploration and grouping                 | Harder to validate; clusters still need business meaning           |
    | **Neural networks / deep learning**          | MLP, CNN, RNN/LSTM, Transformers                                       | Complex nonlinear tasks, images, text, sequences, some tabular cases      | Powerful and flexible                               | More compute, harder debugging, greater reproducibility burden     |
    | **LLM systems**                              | Encoder models, decoder/generative models                              | Language understanding/generation tasks                                   | Strong for unstructured text tasks                  | Cost, latency, hallucinations, evaluation complexity               |
    | **Prompted LLM apps**                        | Prompt-only systems                                                    | Fast LLM prototyping and structured language workflows                    | Low setup cost, quick iteration                     | Brittle prompts, output reliability issues                         |
    | **Fine-tuned LLMs**                          | Instruction-tuned / task-tuned models                                  | Repeated domain-specific language tasks                                   | Better specialization than prompt-only              | More data and training effort                                      |
    | **RAG systems**                              | Retrieval + generation pipelines                                       | Question answering over documents, grounded generation                    | Better grounding than pure prompting                | Retrieval quality becomes critical                                 |
    | **Agent systems**                            | Tool-calling / multi-step agents                                       | Multi-step workflows with decisions and actions                           | Flexible orchestration                              | Easy to overcomplicate; can be fragile and costly                  |
    | **Decision-policy layers on top of models**  | Thresholded classifiers, abstain/escalate policies, calibrated outputs | Business decision systems, risk-sensitive workflows                       | Connects model output to real product behavior      | Requires business-cost thinking, not just metrics                  |



    ## Key principle

    In model engineering, we do not ask "What is the best model?" We ask "Which model family is the right fit for the data, the problem, the cost, the latency, and the deployment constraints?"
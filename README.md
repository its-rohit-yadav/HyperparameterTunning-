# HyperparameterTunning-
# Diabetes Prediction — Neural Network Hyperparameter Tuning

A hands-on exploration of how hyperparameter choices affect a neural network's performance on a binary medical classification task. The dataset is the classic Pima Indians Diabetes Dataset (768 patients, 8 clinical features). The goal is to predict whether a patient is diabetic.

Rather than just training one model and reporting accuracy, this project breaks the tuning process into structured experiments — each one isolates a specific set of hyperparameters so the impact is actually measurable.
## The Experiments

| # | What's tuned | Search Space |
|---|---|---|
| Baseline | Nothing — hand-designed architecture | Fixed: 32 → 16 → 1 |
| Exp 1 | Optimizer | Adam, SGD, RMSprop |
| Exp 2 | Neurons per layer | 8 to 128 (step 16) |
| Exp 3 | Number of hidden layers | 1 to 6 |
| Exp 4 | Everything together | Layers + neurons + activation + optimizer + dropout rate |

Each experiment uses `keras_tuner.RandomSearch` with early stopping to avoid wasting time on configurations that are clearly not working.

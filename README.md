# ADTree: An Alternating Decision Tree Implementation

Alternating Decision Trees differ from decision trees in that they allow for multiple splits at a position on the tree. This achieved by alternating between Prediction Nodes (representing a weight) and a Splitter Node (representing a split), hence the name.

## Features
- Feature Histogramming
- By-feature threading
- Regression and Classification
- Indexed training, rather than propogating full instances
- A 'sampled' training option, tends to be as accurate as 'greedy' training but much faster

## Usage
### Regression
```julia
using .TreeParamaters
using .RegressionTree

params = HyperParameters(
    learning_rate=1.0,
    num_bins=32,
    iterations=100,
    method=:greedy,
    initial_prediction::T = 0.5,
    histogram_version::Symbol = :frequency,
)

my_tree = train(x_train, y_train, params)

my_predictions = predict(my_tree, x_test)
```

### Classification
```julia
using .TreeParamaters
using .MulticlassTree

params = HyperParameters(
    num_classes=3,  # 3 classes
    learning_rate=1.0,
    num_bins=32,
    iterations=100,
    method=:greedy,
    initial_prediction::T = 0.5,
    histogram_version::Symbol = :frequency,
)

my_tree = train(x_train, y_train, params)

my_predictions = predict(my_tree, x_test)
```

## Installation
Currently, install via source and Pkg.jl.

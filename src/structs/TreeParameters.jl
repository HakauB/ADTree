module TreeParameters

using ..HistogramStructs

using Parameters: @with_kw

export HyperParameters, TrainingParameters

"""
    struct HyperParameters

num_classes::Int = 1                            # number of classes, for classification
iterations::Int = 100                           # number of iterations to train
learning_rate::Float = 1.0                      # learning rate
L2::Float = 1.0                                 # L2 regularization
initial_prediction::Float = 0.5                 # initial prediction for a prediction
histogram_version::Symbol = :frequency          # how to histogram the data features
num_bins::Int = 32                              # number of bins for histogramming
case_weights::Vector{Float} || Float = 1.0      # instance weights
loss_func::Function = rmse(y, y_pred)           # loss function
grad_func::Function = gradient(loss_func)       # gradient function (1st order derivative)
hess_func::Function = hessian(loss_func)        # hessian function (2nd order derivative)
method::Symbol = :greedy                        # method to use for training

The user can define their own objective functions via the loss_func, grad_func, and hess_func parameters.
Options for the training method are:
    - :greedy: Greedy training
    - :sampled: Sampled training (train on a sampled subset of paths in the tree)
    - :greedyforest: Greedy forest training (train only on the root node or leaf nodes)
    - :minimizedgreedy: Multiclass greedy training producing only one tree
    - :minimizedsampled: Multiclass sampled training producing only one tree
    - :minimizedgreedyforest: Multiclass greedy forest training producing only one tree
"""
@with_kw struct HyperParameters{T<:AbstractFloat}
    num_classes::Int = 1
    iterations::Int = 100
    learning_rate::T = 1.0
    #L1::T = 0.0
    L2::T = 1.0
    initial_prediction::T = 0.5
    histogram_version::Symbol = :frequency
    num_bins::Int = 32
    case_weights::Union{T, Vector{T}} = 1.0
    loss_func::Function=(y, y_hat) -> sum((y .- y_hat).^2) / size(y, 1)
    grad_func::Function=(y, y_hat) -> y_hat - y
    hess_func::Function=(y, y_hat) -> 1.0
    method::Symbol = :greedy
end

struct TrainingParameters{T<:AbstractFloat}
    hyper_params::HyperParameters{T}
    histogram::HistogramInfo{T}
end

end
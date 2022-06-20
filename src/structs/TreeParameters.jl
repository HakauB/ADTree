module TreeParameters

using ..HistogramStructs

using Parameters: @with_kw

export HyperParameters, TrainingParameters

@with_kw struct HyperParameters{T<:AbstractFloat}
    num_classes::Int = 1
    iterations::Int = 100
    learning_rate::T = 1.0
    L1::T = 0.0
    L2::T = 1.0
    initial_prediction::T = 0.5
    histogram_version::Symbol = :frequency
    num_bins::Int = 30
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
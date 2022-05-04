module TreeUtils

using ..HistogramStructs
using ..TreeParameters
using ..TreeStructs

export initialize_pgh, initialize_pgh, calculate_weight, calculate_gain, vec_sum, vec_sum, compute_bin_gradients, split_dataset, update_pgh!, one_hot, one_hot, calculate_weights, calculate_gains

function initialize_pgh(y::AbstractVector{T}, params::HyperParameters{T}) where T <: AbstractFloat
    return fill(params.initial_prediction, size(y)), zeros(T, size(y)), zeros(T, size(y)) 
end

function calculate_weight(gradient_sum::T, hessian_sum::T, params::HyperParameters{T}) where T <: AbstractFloat
    return -gradient_sum / hessian_sum
end

function calculate_gain(gradient_sum::T, hessian_sum::T, params::HyperParameters{T}) where T <: AbstractFloat
    return gradient_sum^2 / (hessian_sum + params.L2)
end

@inline function vec_sum(x::AbstractVector{T}, indices::Vector{Int}) where T <: Number
    sum = 0.0
    @simd for i in indices
        sum += x[i]
    end
    return sum
    #sum = 0.0
    #@floop for i in eachindex(indices)
    #    @reduce sum += x[indices[i]]
    #end
    #return sum
end

function vec_sum(x::AbstractVector{T}) where T <: AbstractFloat
    return sum(x)
end

@inline function compute_bin_gradients(x::AbstractMatrix{Int}, f::Int, gradients::AbstractVector{T}, hessians::AbstractVector{T}, indices::Vector{Int}, num_bins::Int) where T <: AbstractFloat
    grad_bins = zeros(T, num_bins)
    hess_bins = zeros(T, num_bins)
    @simd for i in indices
        bin_i = x[i, f]
        grad_bins[bin_i] += gradients[i]
        hess_bins[bin_i] += hessians[i]
    end
    return grad_bins, hess_bins
end

function split_dataset(x::AbstractMatrix{Int}, split_index::Int, split_value::Int, indices::Vector{Int})
    left_indices = Vector{Int}()
    right_indices = Vector{Int}()
    @inbounds for i in indices
        if x[i, split_index] <= split_value
            push!(left_indices, i)
        else
            push!(right_indices, i)
        end
    end
    return left_indices, right_indices
end

function update_pgh!(y::AbstractVector{T}, new_weight::T, predictions::AbstractVector{T}, gradients::AbstractVector{T}, hessians::AbstractVector{T}, indices::Vector{Int}, params::HyperParameters) where T <: AbstractFloat
    if isa(params.case_weights, Vector{T})
        @inbounds for i in indices
            predictions[i] += new_weight * params.learning_rate
            gradients[i] = params.grad_func(y[i], predictions[i]) * params.case_weights[i]
            hessians[i] = params.hess_func(y[i], predictions[i]) * params.case_weights[i]
        end
    else
        @inbounds for i in indices
            predictions[i] += new_weight * params.learning_rate
            gradients[i] = params.grad_func(y[i], predictions[i])
            hessians[i] = params.hess_func(y[i], predictions[i])
        end
    end
end

###############################################################################
#MULTI-CLASS###################################################################

function one_hot(y::Int, num_classes::Int)
    v = zeros(Int, num_classes)
    v[y] = 1
    return v
end

function one_hot(y::AbstractVector{Int}, num_classes::Int)
    v = zeros(Int, size(y, 1), num_classes)
    for i in 1:size(y, 1)
        v[i, y[i]] = 1
    end
    return v
end

function initialize_pgh(y::AbstractMatrix{Int}, params::HyperParameters{T}) where T <: AbstractFloat
    if params.num_classes > 2 && params.initial_prediction != 0.5
        return fill(params.initial_prediction, size(y)), zeros(T, size(y)), zeros(T, size(y))
    end
    return fill(1.0 / size(y, 2), size(y)), zeros(T, size(y)), zeros(T, size(y)) 
end

function calculate_weights(gradient_sums::AbstractVector{T}, hessian_sums::AbstractVector{T}, params::HyperParameters{T}) where T <: AbstractFloat
    return -gradient_sums ./ hessian_sums
end

function calculate_gains(gradient_sums::AbstractVector{T}, hessian_sums::AbstractVector{T}, params::HyperParameters{T}) where T <: AbstractFloat
    return gradient_sums.^2 ./ (hessian_sums .+ params.L2)
end

function vec_sum(x::AbstractMatrix{T}, f::Int, indices::Vector{Int}) where T <: AbstractFloat
    sum = 0.0
    @inbounds for i in indices
        sum = sum + x[i, f]
    end
    return sum
end

@inline function compute_bin_gradients(x::AbstractMatrix{Int}, f::Int, gradients::AbstractMatrix{T}, hessians::AbstractMatrix{T}, y_i::Int, indices::Vector{Int}, num_bins::Int) where T <: AbstractFloat
    grad_bins = zeros(T, num_bins)
    hess_bins = zeros(T, num_bins)
    @simd for i in indices
        bin_i = x[i, f]
        grad_bins[bin_i] += gradients[i, y_i]
        hess_bins[bin_i] += hessians[i, y_i]
    end
    return grad_bins, hess_bins
end

function update_pgh!(y::AbstractMatrix{Int}, new_weight::T, predictions::AbstractMatrix{T}, gradients::AbstractMatrix{T}, hessians::AbstractMatrix{T}, y_i::Int, indices::Vector{Int}, params::HyperParameters) where T <: Number
    if isa(params.case_weights, Vector{T})
        predictions[indices, y_i] .= predictions[indices, y_i] .+ new_weight .* params.learning_rate
        gradients[indices, :] .= params.grad_func(y[indices, :], predictions[indices, :]) .* params.case_weights[indices]
        hessians[indices, :] .= params.hess_func(y[indices, :], predictions[indices, :]) .* params.case_weights[indices]
    else
        predictions[indices, y_i] .= predictions[indices, y_i] .+ new_weight .* params.learning_rate
        gradients[indices, :] .= params.grad_func(y[indices, :], predictions[indices, :])
        hessians[indices, :] .= params.hess_func(y[indices, :], predictions[indices, :])
    end
end

###############################################################################

end
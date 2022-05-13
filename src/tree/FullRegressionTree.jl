module FullRegressionTree

using ..HistogramStructs
using ..TreeStructs
using ..TreeParameters
using ..Histogram
using ..TreeUtils

using StatsBase

export train

struct FullSplitterNode <: Main.TreeStructs.AbstractSplitterNode
    split_feature::Int
    split_value::AbstractFloat
    left_node::AbstractPredictionNode
    right_node::AbstractPredictionNode
end

struct FullPredictionNode{T<:AbstractFloat} <: Main.TreeStructs.AbstractPredictionNode
    id::Int
    weight::T
    gain::T
    indices::Vector{Int}
    splitter_nodes::Vector{FullSplitterNode}
    class_index::Int
end

function find_best_split(x::AbstractMatrix{T}, gradients::AbstractVector{T}, hessians::AbstractVector{T}, indices::AbstractVector{Int}, params::HyperParameters) where T <: AbstractFloat
    best_gain = -Inf
    best_split_index = 0
    best_split_value = 0.0

    subset_x = x[indices, :]
    subset_gradients = gradients[indices]
    subset_hessians = hessians[indices]

    for f in 1:size(x, 2)
        right_grad_sum = sum(subset_gradients)
        right_hess_sum = sum(subset_hessians)
        left_grad_sum = 0.0
        left_hess_sum = 0.0

        sort_indices = sortperm(subset_x[:, f])
        for i in 1:size(sort_indices, 1) - 1
            left_grad_sum += subset_gradients[sort_indices[i]]
            left_hess_sum += subset_hessians[sort_indices[i]]
            right_grad_sum -= subset_gradients[sort_indices[i]]
            right_hess_sum -= subset_hessians[sort_indices[i]]

            if subset_x[sort_indices[i], f] == subset_x[sort_indices[i + 1], f]
                continue
            end

            gain = calculate_gain(left_grad_sum, left_hess_sum, params) + calculate_gain(right_grad_sum, right_hess_sum, params)
            if gain > best_gain
                best_gain = gain
                best_split_index = f
                best_split_value = (subset_x[sort_indices[i], f] + subset_x[sort_indices[i + 1], f]) / 2.0
            end
        end
    end
    return best_gain, best_split_index, best_split_value
end

function split_dataset(x::AbstractMatrix{T}, best_split_index::Int, best_split_value::T, indices::AbstractVector{Int}, params::HyperParameters) where T <: AbstractFloat
    left_indices = Vector{Int}()
    right_indices = Vector{Int}()
    for ind in indices
        if x[ind, best_split_index] <= best_split_value
            push!(left_indices, ind)
        else
            push!(right_indices, ind)
        end
    end
    return left_indices, right_indices
end

function train_greedy(x::AbstractMatrix{T}, y::AbstractVector{T}, params::HyperParameters) where T <: AbstractFloat
    predictions, gradients, hessians = initialize_pgh(y, params)
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights
    
    grad_sum, hess_sum = sum(gradients), sum(hessians)
    root_weight = calculate_weight(grad_sum, hess_sum, params)
    root_gain = calculate_gain(grad_sum, hess_sum, params)
    root = FullPredictionNode(1, root_weight, root_gain, collect(1:size(x, 1)), Vector{FullSplitterNode}(), 1)
    predictions .+= root_weight * params.learning_rate
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights

    stack = Vector{AbstractPredictionNode}()
    push!(stack, root)

    for iter in 1:params.iterations
        best_node_index = 0
        best_gain = -Inf
        best_split_index = 0
        best_split_value = 0.0

        for i in 1:size(stack, 1)
            node = stack[i]
            gain, split_index, split_value = find_best_split(x, gradients, hessians, node.indices, params)
            if gain > best_gain
                best_node_index = i
                best_gain = gain
                best_split_index = split_index
                best_split_value = split_value
            end
        end
        best_node = stack[best_node_index]

        left_indices, right_indices = split_dataset(x, best_split_index, best_split_value, best_node.indices, params)
        left_g = sum(gradients[left_indices])
        left_h = sum(hessians[left_indices])
        right_g = sum(gradients[right_indices])
        right_h = sum(hessians[right_indices])

        left_weight = calculate_weight(left_g, left_h, params)
        right_weight = calculate_weight(right_g, right_h, params)
        left_gain = calculate_gain(left_g, left_h, params)
        right_gain = calculate_gain(right_g, right_h, params)

        left_node = FullPredictionNode(2 * iter, left_weight, left_gain, left_indices, Vector{FullSplitterNode}(), 1)
        right_node = FullPredictionNode(2 * iter + 1, right_weight, right_gain, right_indices, Vector{FullSplitterNode}(), 1)
        new_splitter = FullSplitterNode(best_split_index, best_split_value, left_node, right_node)

        push!(best_node.splitter_nodes, new_splitter)
        push!(stack, left_node)
        push!(stack, right_node)

        predictions[left_indices] .+= left_weight * params.learning_rate
        gradients[left_indices] .= params.grad_func(y[left_indices], predictions[left_indices]) .* params.case_weights
        hessians[left_indices] .= params.hess_func(y[left_indices], predictions[left_indices]) .* params.case_weights

        predictions[right_indices] .+= right_weight * params.learning_rate
        gradients[right_indices] .= params.grad_func(y[right_indices], predictions[right_indices]) .* params.case_weights
        hessians[right_indices] .= params.hess_func(y[right_indices], predictions[right_indices]) .* params.case_weights

        println("Best split: ", best_split_index, " @ ", best_split_value)
        println(sqrt(params.loss_func(y, predictions)))
    end
    return root
end

function train(x::AbstractMatrix{T}, y::AbstractVector{T}, params::HyperParameters; method::Symbol=:greedy) where T <: AbstractFloat
    if method == :greedy
        return train_greedy(x, y, params)
    else
        error("Unknown method: " + method)
    end
end

function predict(root::FullPredictionNode, x::AbstractMatrix{T}) where T <: AbstractFloat
    predictions = fill(0.5, size(x, 1))
    for i in 1:size(x, 1)
        nodes = [root]
        for node in nodes
            predictions[i] += node.weight
            for splitter in node.splitter_nodes
                if x[i, splitter.split_feature] <= splitter.split_value
                    push!(nodes, splitter.left_node)
                else
                    push!(nodes, splitter.right_node)
                end
            end
        end
    end
    return predictions
end
    
end
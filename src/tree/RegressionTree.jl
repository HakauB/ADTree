module RegressionTree

using ..HistogramStructs
using ..TreeStructs
using ..TreeParameters
using ..Histogram
using ..TreeUtils

using StatsBase

export train, predict, clear_train_data!, treesize

function bin_split(x::AbstractMatrix{Int}, gradients::AbstractVector{T}, hessians::AbstractVector{T}, indices::Vector{Int}, bin_sizes::Vector{Int}, params::HyperParameters) where T <: AbstractFloat
    search_best_gain = 0.0
    search_best_split_index = 0
    search_best_split_value = 0.0
    split_search_results = Vector{Tuple{T, Int, Int}}(undef, size(x, 2))
    Threads.@threads for f in 1:size(x, 2)
        if bin_sizes[f] == 1
            split_search_results[f] = (0.0, 0, 0)
            continue
        end
        gradient_bins, hessian_bins = compute_bin_gradients(x, f, gradients, hessians, indices, bin_sizes[f])
        gradient_bin_sums = cumsum(gradient_bins, dims=1)
        hessian_bin_sums = cumsum(hessian_bins, dims=1)
        gradient_sum = gradient_bin_sums[end]
        hessian_sum = hessian_bin_sums[end]

        gains = zeros(T, size(gradient_bin_sums, 1) - 1)
        @inbounds for i in 1:size(gains, 1)
            left = calculate_gain(gradient_bin_sums[i], hessian_bin_sums[i], params)
            right = calculate_gain(gradient_sum - gradient_bin_sums[i], hessian_sum - hessian_bin_sums[i], params)
            gains[i] = left + right
            if isnan(gains[i])
                gains[i] = 0.0
            end
        end

        best_gain_index = argmax(gains)
        best_gain = gains[best_gain_index]
        best_split = best_gain_index
        split_search_results[f] = (best_gain, f, best_split)
    end
    search_best_gain, search_best_split_index, search_best_split_value =  maximum(split_search_results)
    return search_best_gain, search_best_split_index, search_best_split_value
end

function train_greedy(x::AbstractMatrix{Int}, y::AbstractVector{T}, training_params::TrainingParameters) where T <: AbstractFloat
    params = training_params.hyper_params
    bin_sizes = training_params.histogram.num_bins
    predictions, gradients, hessians = initialize_pgh(y, params)
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights
    
    grad_sum, hess_sum = sum(gradients), sum(hessians)
    root_weight = calculate_weight(grad_sum, hess_sum, params)
    root_gain = calculate_gain(grad_sum, hess_sum, params)
    root = PredictionNode(1, root_weight, root_gain, collect(1:size(x, 1)), Vector{SplitterNode}(), 1)
    predictions .+= root_weight * params.learning_rate
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights

    stack = Vector{AbstractPredictionNode}()
    push!(stack, root)

    for iter in 1:params.iterations
        best_node_index = 0
        best_gain = 0.0
        best_split_index = 0
        best_split_value = 0
        for i in 1:size(stack, 1)
            node = stack[i]
            gain, split_index, split_value = bin_split(x, gradients, hessians, node.indices, bin_sizes, params)
            if gain > best_gain
                best_node_index = i
                best_gain = gain
                best_split_index = split_index
                best_split_value = split_value
            end
        end
        best_node = stack[best_node_index]
        
        left_inds, right_inds = split_dataset(x, best_split_index, best_split_value, best_node.indices)
        
        left_g, left_h = vec_sum(gradients, left_inds), vec_sum(hessians, left_inds)
        right_g, right_h = vec_sum(gradients, right_inds), vec_sum(hessians, right_inds)
        left_weight = calculate_weight(left_g, left_h, params)
        left_gain = calculate_gain(left_g, left_h, params)
        right_weight = calculate_weight(right_g, right_h, params)
        right_gain = calculate_gain(right_g, right_h, params)
        
        left_node = PredictionNode(iter * 2, left_weight, left_gain, left_inds, Vector{SplitterNode}(), 1)
        right_node = PredictionNode(iter * 2 + 1, right_weight, right_gain, right_inds, Vector{SplitterNode}(), 1)
        new_splitter = SplitterNode(best_split_index, best_split_value, left_node, right_node)
        push!(best_node.splitter_nodes, new_splitter)
        push!(stack, left_node)
        push!(stack, right_node)
        
        update_pgh!(y, left_weight, predictions, gradients, hessians, left_inds, params)
        update_pgh!(y, right_weight, predictions, gradients, hessians, right_inds, params)
    end
    return root
end

function train_greedy_forest(x::AbstractMatrix{Int}, y::AbstractVector{T}, training_params::TrainingParameters) where T <: AbstractFloat
    params = training_params.hyper_params
    bin_sizes = training_params.histogram.num_bins
    predictions, gradients, hessians = initialize_pgh(y, params)
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights
    
    grad_sum, hess_sum = sum(gradients), sum(hessians)
    root_weight = calculate_weight(grad_sum, hess_sum, params)
    root_gain = calculate_gain(grad_sum, hess_sum, params)
    root = PredictionNode(1, root_weight, root_gain, collect(1:size(x, 1)), Vector{SplitterNode}(), 1)
    predictions .+= root_weight * params.learning_rate
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights

    stack = Vector{AbstractPredictionNode}()
    push!(stack, root)

    for iter in 1:params.iterations
        best_node_index = 0
        best_gain = 0.0
        best_split_index = 0
        best_split_value = 0
        for i in 1:size(stack, 1)
            node = stack[i]
            gain, split_index, split_value = bin_split(x, gradients, hessians, node.indices, bin_sizes, params)
            if gain > best_gain
                best_node_index = i
                best_gain = gain
                best_split_index = split_index
                best_split_value = split_value
            end
        end
        #best_node = stack[best_node_index]
        best_node = popat!(stack, best_node_index)
        
        left_inds, right_inds = split_dataset(x, best_split_index, best_split_value, best_node.indices)
        
        left_g, left_h = vec_sum(gradients, left_inds), vec_sum(hessians, left_inds)
        right_g, right_h = vec_sum(gradients, right_inds), vec_sum(hessians, right_inds)
        left_weight = calculate_weight(left_g, left_h, params)
        left_gain = calculate_gain(left_g, left_h, params)
        right_weight = calculate_weight(right_g, right_h, params)
        right_gain = calculate_gain(right_g, right_h, params)
        
        left_node = PredictionNode(iter * 2, left_weight, left_gain, left_inds, Vector{SplitterNode}(), 1)
        right_node = PredictionNode(iter * 2 + 1, right_weight, right_gain, right_inds, Vector{SplitterNode}(), 1)
        new_splitter = SplitterNode(best_split_index, best_split_value, left_node, right_node)
        push!(best_node.splitter_nodes, new_splitter)
        push!(stack, left_node)
        push!(stack, right_node)

        if best_node == root
            pushfirst!(stack, best_node)
        end
        
        update_pgh!(y, left_weight, predictions, gradients, hessians, left_inds, params)
        update_pgh!(y, right_weight, predictions, gradients, hessians, right_inds, params)
    end
    return root
end

function train_sampled(x::AbstractMatrix{Int}, y::AbstractVector{T}, training_params::TrainingParameters) where T <: AbstractFloat
    params = training_params.hyper_params
    bin_sizes = training_params.histogram.num_bins

    predictions, gradients, hessians = initialize_pgh(y, params)
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights
    
    grad_sum, hess_sum = sum(gradients), sum(hessians)
    root_weight = calculate_weight(grad_sum, hess_sum, params)
    root_gain = calculate_gain(grad_sum, hess_sum, params)
    root = PredictionNode(1, root_weight, root_gain, collect(1:size(x, 1)), Vector{SplitterNode}(), 1)
    predictions .+= root_weight * params.learning_rate
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights

    stack = Vector{AbstractPredictionNode}()
    push!(stack, root)

    for iter in 1:params.iterations
        node_indices = collect(2:size(stack, 1))
        if iter > 6
            sampling_k = Int(floor(sqrt(iter)))
            node_indices = sample(node_indices, sampling_k; replace=false)
        end
        push!(node_indices, 1)
        best_node_index = 0
        best_gain = 0.0
        best_split_index = 0
        best_split_value = 0
        for i in 1:size(node_indices, 1)
            node = stack[node_indices[i]]
            gain, split_index, split_value = bin_split(x, gradients, hessians, node.indices, bin_sizes, params)
            if gain > best_gain
                best_node_index = i
                best_gain = gain
                best_split_index = split_index
                best_split_value = split_value
            end
        end
        best_node = stack[node_indices[best_node_index]]
        
        left_inds, right_inds = split_dataset(x, best_split_index, best_split_value, best_node.indices)
        
        left_g, left_h = vec_sum(gradients, left_inds), vec_sum(hessians, left_inds)
        right_g, right_h = vec_sum(gradients, right_inds), vec_sum(hessians, right_inds)
        left_weight = calculate_weight(left_g, left_h, params)
        left_gain = calculate_gain(left_g, left_h, params)
        right_weight = calculate_weight(right_g, right_h, params)
        right_gain = calculate_gain(right_g, right_h, params)
        
        left_node = PredictionNode(iter * 2, left_weight, left_gain, left_inds, Vector{SplitterNode}(), 1)
        right_node = PredictionNode(iter * 2 + 1, right_weight, right_gain, right_inds, Vector{SplitterNode}(), 1)
        new_splitter = SplitterNode(best_split_index, best_split_value, left_node, right_node)
        push!(best_node.splitter_nodes, new_splitter)
        push!(stack, left_node)
        push!(stack, right_node)
        
        update_pgh!(y, left_weight, predictions, gradients, hessians, left_inds, params)
        update_pgh!(y, right_weight, predictions, gradients, hessians, right_inds, params)
    end
    return root
end

function train_random(x::AbstractMatrix{Int}, y::AbstractVector{T}, training_params::TrainingParameters) where T <: AbstractFloat
    params = training_params.hyper_params
    bin_sizes = training_params.histogram.num_bins
    predictions, gradients, hessians = initialize_pgh(y, params)
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights
    
    grad_sum, hess_sum = sum(gradients), sum(hessians)
    root_weight = calculate_weight(grad_sum, hess_sum, params)
    root_gain = calculate_gain(grad_sum, hess_sum, params)
    root = PredictionNode(1, root_weight, root_gain, collect(1:size(x, 1)), Vector{SplitterNode}(), 1)
    predictions .+= root_weight * params.learning_rate
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights

    stack = Vector{AbstractPredictionNode}()
    push!(stack, root)

    for iter in 1:params.iterations
        best_node_index = 0
        best_gain = 0.0
        best_split_index = 0
        best_split_value = 0

        node_index = rand(1:size(stack, 1))
        node = stack[node_index]
        gain, split_index, split_value = bin_split(x, gradients, hessians, node.indices, bin_sizes, params)
        if gain > best_gain
            best_node_index = node_index
            best_gain = gain
            best_split_index = split_index
            best_split_value = split_value
        else
            #TODO: This is a hack to make sure we don't get stuck in a local minimum. Fix this?
            attempts = 0
            while attempts < 10
                node_index = rand(1:size(stack, 1))
                node = stack[node_index]
                gain, split_index, split_value = bin_split(x, gradients, hessians, node.indices, bin_sizes, params)
                attempts += 1
                if gain > best_gain
                    best_node_index = node_index
                    best_gain = gain
                    best_split_index = split_index
                    best_split_value = split_value
                    break                    
                end
            end
            if attempts == 10
                println("Couldn't find a good split after 10 attempts")
                break
            end
        end
        best_node = stack[best_node_index]
        
        left_inds, right_inds = split_dataset(x, best_split_index, best_split_value, best_node.indices)
        
        left_g, left_h = vec_sum(gradients, left_inds), vec_sum(hessians, left_inds)
        right_g, right_h = vec_sum(gradients, right_inds), vec_sum(hessians, right_inds)
        left_weight = calculate_weight(left_g, left_h, params)
        left_gain = calculate_gain(left_g, left_h, params)
        right_weight = calculate_weight(right_g, right_h, params)
        right_gain = calculate_gain(right_g, right_h, params)
        
        left_node = PredictionNode(iter * 2, left_weight, left_gain, left_inds, Vector{SplitterNode}(), 1)
        right_node = PredictionNode(iter * 2 + 1, right_weight, right_gain, right_inds, Vector{SplitterNode}(), 1)
        new_splitter = SplitterNode(best_split_index, best_split_value, left_node, right_node)
        push!(best_node.splitter_nodes, new_splitter)
        push!(stack, left_node)
        push!(stack, right_node)
        
        update_pgh!(y, left_weight, predictions, gradients, hessians, left_inds, params)
        update_pgh!(y, right_weight, predictions, gradients, hessians, right_inds, params)
    end
    println(sqrt(params.loss_func(y, predictions)))
    return root
end

"""
    train(x::AbstractMatrix{T}, y::AbstractVector{T}, params::HyperParameters) where T <: AbstractFloat

Train a tree using the given data, (x, y), according to the provided hyper parameters.
Returns the tree (root).
"""
function train(x::AbstractMatrix{T}, y::AbstractVector{T}, params::HyperParameters) where T <: AbstractFloat
    x_binned, hist_info = histogram(x, params)
    training_params = TrainingParameters(params, hist_info)
    if params.method == :greedy
        root = train_greedy(x_binned, y, training_params)
        return Tree(root, training_params)
    elseif params.method == :sampled
        root = train_sampled(x_binned, y, training_params)
        return Tree(root, training_params)
    elseif params.method == :random
        root = train_random(x_binned, y, training_params)
        return Tree(root, training_params)
    elseif params.method == :greedyforest
        root = train_greedy_forest(x_binned, y, training_params)
        return Tree(root, training_params)
    else
        error("Unknown training method: " + params.method)
    end
end

function predict(tree::Tree, x::AbstractVector{T}) where T <: AbstractFloat
    x_binned = binindex_column(x, tree.training_params.histogram.hist_limits)
    prediction = tree.training_params.hyper_params.initial_prediction
    nodes = [tree.root]
    for curr_node in nodes
        prediction += curr_node.weight * tree.training_params.hyper_params.learning_rate
        for splitter in curr_node.splitter_nodes
            if x_binned[splitter.split_feature] <= splitter.split_value
                push!(nodes, splitter.left_node)
            else
                push!(nodes, splitter.right_node)
            end
        end
    end
    return prediction
end

"""
    predict(tree::Tree, x::AbstractMatrix{T}) where T <: AbstractFloat
    
Calculates the predictions for the given data x, using the given trained tree.
Returns a vector of predictions.
"""
function predict(tree::Tree, x::AbstractMatrix{T}) where T <: AbstractFloat
    predictions = fill(tree.training_params.hyper_params.initial_prediction, size(x, 1))
    x_binned, hist_info = histogram(x, tree.training_params.histogram)
    for i in 1:size(x, 1)
        nodes = [tree.root]
        for curr_node in nodes
            predictions[i] += curr_node.weight * tree.training_params.hyper_params.learning_rate
            for splitter in curr_node.splitter_nodes
                if x_binned[i, splitter.split_feature] <= splitter.split_value
                    push!(nodes, splitter.left_node)
                else
                    push!(nodes, splitter.right_node)
                end
            end
        end
    end
    return predictions
end

function _clear_train_data(node::AbstractPredictionNode)
    nodes = [node]
    for curr_node in nodes
        for ind in 1:size(curr_node.indices, 1)
            pop!(curr_node.indices)
        end
        for splitter in curr_node.splitter_nodes
            push!(nodes, splitter.left_node)
            push!(nodes, splitter.right_node)   
        end
    end
end

"""
    clear_train_data!(tree::Tree)
Clears the training indices of the given tree, significantly reducing the in-memory size.
Returns the tree.
"""
function clear_train_data!(tree::Tree)
    _clear_train_data(tree.root)
    return tree
end

"""
    treesize(tree::Tree)
Returns the size of the given tree.
"""
function treesize(tree::Tree)
    # This is not an exposed function and may not be future-proof.
    return Base.summarysize(tree)
end

end
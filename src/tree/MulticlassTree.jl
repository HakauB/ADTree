module MulticlassTree

using ..HistogramStructs
using ..TreeStructs
using ..TreeParameters
using ..Histogram
using ..TreeUtils

using StatsBase

export train, predict

function bin_split(x::AbstractMatrix{Int}, gradients::AbstractMatrix{T}, hessians::AbstractMatrix{T}, y_i::Int, indices::Vector{Int}, bin_sizes::Vector{Int}, params::HyperParameters) where T <: AbstractFloat
    search_best_gain = 0.0
    search_best_split_index = 0
    search_best_split_value = 0.0
    split_search_results = Vector{Tuple{T, Int, Int}}(undef, size(x, 2))
    Threads.@threads for f in 1:size(x, 2)
        gradient_bins, hessian_bins = compute_bin_gradients(x, f, gradients, hessians, y_i, indices, bin_sizes[f])
        gradient_bin_sums = cumsum(gradient_bins, dims=1)
        hessian_bin_sums = cumsum(hessian_bins, dims=1)
        gradient_sum = gradient_bin_sums[end]
        hessian_sum = hessian_bin_sums[end]

        gains = zeros(size(gradient_bin_sums))
        @inbounds for i in eachindex(gradient_bin_sums, hessian_bin_sums)
            gains[i] = calculate_gain(gradient_bin_sums[i], hessian_bin_sums[i], params) + calculate_gain(gradient_sum - gradient_bin_sums[i], hessian_sum - hessian_bin_sums[i], params)
            if isnan(gains[i])
                gains[i] = 0.0
            end
        end
        pop!(gains)
        best_gain_index = argmax(gains)
        best_gain = gains[best_gain_index]
        best_split = best_gain_index
        split_search_results[f] = (best_gain, f, best_split)
    end
    search_best_gain, search_best_split_index, search_best_split_value =  maximum(split_search_results)
    return search_best_gain, search_best_split_index, search_best_split_value
end

function train_greedy(x::AbstractMatrix{Int}, y::AbstractMatrix{Int}, training_params::TrainingParameters)
    params = training_params.hyper_params
    bin_sizes = training_params.histogram.num_bins

    predictions, gradients, hessians = initialize_pgh(y, params)
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights
    
    roots = Vector{AbstractPredictionNode}()
    for i in 1:params.num_classes
        grad_sum = sum(gradients[:, i])
        hess_sum = sum(hessians[:, i])
        root_weight = calculate_weight(grad_sum, hess_sum, params)
        root_gain = calculate_gain(grad_sum, hess_sum, params)
        curr_root = PredictionNode(1, root_weight, root_gain, collect(1:size(x, 1)), Vector{SplitterNode}(), i)
        push!(roots, curr_root)
        predictions[:, i] .+= root_weight .* params.learning_rate
    end

    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights

    stacks = Vector{Vector{AbstractPredictionNode}}()
    for i in 1:params.num_classes
        new_stack = Vector{AbstractPredictionNode}()
        push!(new_stack, roots[i])
        push!(stacks, new_stack)
    end

    for iter in 1:params.iterations
        for y_i in 1:params.num_classes
            best_node_index = 0
            best_gain = 0.0
            best_split_index = 0
            best_split_value = 0
            curr_stack = stacks[y_i]
            for i in 1:size(curr_stack, 1)
                node = curr_stack[i]
                gain, split_index, split_value = bin_split(x, gradients, hessians, y_i, node.indices, bin_sizes, params)
                if gain > best_gain
                    best_node_index = i
                    best_gain = gain
                    best_split_index = split_index
                    best_split_value = split_value
                end
            end
            best_node = curr_stack[best_node_index]

            left_inds, right_inds = split_dataset(x, best_split_index, best_split_value, best_node.indices)

            left_g, left_h = vec_sum(gradients, y_i, left_inds), vec_sum(hessians, y_i, left_inds)
            right_g, right_h = vec_sum(gradients, y_i, right_inds), vec_sum(hessians, y_i, right_inds)

            left_weight = calculate_weight(left_g, left_h, params)
            right_weight = calculate_weight(right_g, right_h, params)
            left_gain = calculate_gain(left_g, left_h, params)
            right_gain = calculate_gain(right_g, right_h, params)

            left_node = PredictionNode(best_node.id * 2, left_weight, left_gain, collect(left_inds), Vector{SplitterNode}(), y_i)
            right_node = PredictionNode(best_node.id * 2 + 1, right_weight, right_gain, collect(right_inds), Vector{SplitterNode}(), y_i)
            new_splitter = SplitterNode(best_split_index, best_split_value, left_node, right_node)
            push!(best_node.splitter_nodes, new_splitter)
            push!(curr_stack, left_node)
            push!(curr_stack, right_node)

            update_pgh!(y, left_weight, predictions, gradients, hessians, y_i, left_inds, params)
            update_pgh!(y, right_weight, predictions, gradients, hessians, y_i, right_inds, params)
        end
    end
    return roots
end

function train_greedy_forest(x::AbstractMatrix{Int}, y::AbstractMatrix{Int}, training_params::TrainingParameters)
    params = training_params.hyper_params
    bin_sizes = training_params.histogram.num_bins

    predictions, gradients, hessians = initialize_pgh(y, params)
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights
    
    roots = Vector{AbstractPredictionNode}()
    for i in 1:params.num_classes
        grad_sum = sum(gradients[:, i])
        hess_sum = sum(hessians[:, i])
        root_weight = calculate_weight(grad_sum, hess_sum, params)
        root_gain = calculate_gain(grad_sum, hess_sum, params)
        curr_root = PredictionNode(1, root_weight, root_gain, collect(1:size(x, 1)), Vector{SplitterNode}(), i)
        push!(roots, curr_root)
        predictions[:, i] .+= root_weight .* params.learning_rate
    end

    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights

    stacks = Vector{Vector{AbstractPredictionNode}}()
    for i in 1:params.num_classes
        new_stack = Vector{AbstractPredictionNode}()
        push!(new_stack, roots[i])
        push!(stacks, new_stack)
    end

    for iter in 1:params.iterations
        for y_i in 1:params.num_classes
            best_node_index = 0
            best_gain = 0.0
            best_split_index = 0
            best_split_value = 0
            curr_stack = stacks[y_i]
            for i in 1:size(curr_stack, 1)
                node = curr_stack[i]
                gain, split_index, split_value = bin_split(x, gradients, hessians, y_i, node.indices, bin_sizes, params)
                if gain > best_gain
                    best_node_index = i
                    best_gain = gain
                    best_split_index = split_index
                    best_split_value = split_value
                end
            end
            best_node = popat!(curr_stack, best_node_index)

            left_inds, right_inds = split_dataset(x, best_split_index, best_split_value, best_node.indices)

            left_g, left_h = vec_sum(gradients, y_i, left_inds), vec_sum(hessians, y_i, left_inds)
            right_g, right_h = vec_sum(gradients, y_i, right_inds), vec_sum(hessians, y_i, right_inds)

            left_weight = calculate_weight(left_g, left_h, params)
            right_weight = calculate_weight(right_g, right_h, params)
            left_gain = calculate_gain(left_g, left_h, params)
            right_gain = calculate_gain(right_g, right_h, params)

            left_node = PredictionNode(best_node.id * 2, left_weight, left_gain, collect(left_inds), Vector{SplitterNode}(), y_i)
            right_node = PredictionNode(best_node.id * 2 + 1, right_weight, right_gain, collect(right_inds), Vector{SplitterNode}(), y_i)
            new_splitter = SplitterNode(best_split_index, best_split_value, left_node, right_node)
            push!(best_node.splitter_nodes, new_splitter)
            push!(curr_stack, left_node)
            push!(curr_stack, right_node)

            if best_node == roots[y_i]
                pushfirst!(curr_stack, best_node)
            end

            update_pgh!(y, left_weight, predictions, gradients, hessians, y_i, left_inds, params)
            update_pgh!(y, right_weight, predictions, gradients, hessians, y_i, right_inds, params)
        end
    end
    return roots
end

function train_sampled(x::AbstractMatrix{Int}, y::AbstractMatrix{Int}, training_params::TrainingParameters) where T <: AbstractFloat
    params = training_params.hyper_params
    bin_sizes = training_params.histogram.num_bins

    predictions, gradients, hessians = initialize_pgh(y, params)
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights
    
    roots = Vector{AbstractPredictionNode}()
    for i in 1:params.num_classes
        grad_sum = sum(gradients[:, i])
        hess_sum = sum(hessians[:, i])
        root_weight = calculate_weight(grad_sum, hess_sum, params)
        root_gain = calculate_gain(grad_sum, hess_sum, params)
        curr_root = PredictionNode(1, root_weight, root_gain, collect(1:size(x, 1)), Vector{SplitterNode}(), i)
        push!(roots, curr_root)
        predictions[:, i] .+= root_weight .* params.learning_rate
    end

    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights

    stacks = Vector{Vector{AbstractPredictionNode}}()
    for i in 1:params.num_classes
        new_stack = Vector{AbstractPredictionNode}()
        push!(new_stack, roots[i])
        push!(stacks, new_stack)
    end

    for iter in 1:params.iterations
        for y_i in 1:params.num_classes
            curr_stack = stacks[y_i]
            node_indices = collect(2:size(curr_stack, 1))
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
                node = curr_stack[node_indices[i]]
                gain, split_index, split_value = bin_split(x, gradients, hessians, y_i, node.indices, bin_sizes, params)
                if gain > best_gain
                    best_node_index = i
                    best_gain = gain
                    best_split_index = split_index
                    best_split_value = split_value
                end
            end
            best_node = curr_stack[node_indices[best_node_index]]

            left_inds, right_inds = split_dataset(x, best_split_index, best_split_value, best_node.indices)

            left_g, left_h = vec_sum(gradients, y_i, left_inds), vec_sum(hessians, y_i, left_inds)
            right_g, right_h = vec_sum(gradients, y_i, right_inds), vec_sum(hessians, y_i, right_inds)

            left_weight = calculate_weight(left_g, left_h, params)
            right_weight = calculate_weight(right_g, right_h, params)
            left_gain = calculate_gain(left_g, left_h, params)
            right_gain = calculate_gain(right_g, right_h, params)

            left_node = PredictionNode(best_node.id * 2, left_weight, left_gain, collect(left_inds), Vector{SplitterNode}(), y_i)
            right_node = PredictionNode(best_node.id * 2 + 1, right_weight, right_gain, collect(right_inds), Vector{SplitterNode}(), y_i)
            new_splitter = SplitterNode(best_split_index, best_split_value, left_node, right_node)
            push!(best_node.splitter_nodes, new_splitter)
            push!(curr_stack, left_node)
            push!(curr_stack, right_node)

            update_pgh!(y, left_weight, predictions, gradients, hessians, y_i, left_inds, params)
            update_pgh!(y, right_weight, predictions, gradients, hessians, y_i, right_inds, params)
        end
    end
    return roots
end

function train_minimized_greedy(x::AbstractMatrix{Int}, y::AbstractMatrix{Int}, training_params::TrainingParameters) where T <: AbstractFloat
    params = training_params.hyper_params
    bin_sizes = training_params.histogram.num_bins

    predictions, gradients, hessians = initialize_pgh(y, params)
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights

    root_weight = 0.0
    root_gain = 0.0
    root_y_i = 0
    for y_i in 1:params.num_classes
        grad_sum = vec_sum(gradients, y_i, collect(1:size(x, 1)))
        hess_sum = vec_sum(hessians, y_i, collect(1:size(x, 1)))
        weight = calculate_weight(grad_sum, hess_sum, params)
        gain = calculate_gain(grad_sum, hess_sum, params)
        if gain > root_gain
            root_weight = weight
            root_gain = gain
            root_y_i = y_i
        end
    end
    root = PredictionNode(1, root_weight, root_gain, collect(1:size(x, 1)), Vector{SplitterNode}(), root_y_i)
    predictions[:, root_y_i] .+= root_weight .* params.learning_rate
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights

    stack = Vector{AbstractPredictionNode}()
    push!(stack, root)

    for iter in 1:params.iterations
        best_node_index = 0
        best_gain = 0.0
        best_split_index = 0
        best_split_value = 0
        best_y_i = 0
        for i in 1:size(stack, 1)
            node = stack[i]
            for y_i in 1:params.num_classes
                gain, split_index, split_value = bin_split(x, gradients, hessians, y_i, node.indices, bin_sizes, params)
                if gain > best_gain
                    best_node_index = i
                    best_gain = gain
                    best_split_index = split_index
                    best_split_value = split_value
                    best_y_i = y_i
                end
            end
        end
        best_node = stack[best_node_index]
        left_inds, right_inds = split_dataset(x, best_split_index, best_split_value, best_node.indices)

        left_g, left_h = vec_sum(gradients, best_y_i, left_inds), vec_sum(hessians, best_y_i, left_inds)
        right_g, right_h = vec_sum(gradients, best_y_i, right_inds), vec_sum(hessians, best_y_i, right_inds)

        left_weight = calculate_weight(left_g, left_h, params)
        right_weight = calculate_weight(right_g, right_h, params)
        left_gain = calculate_gain(left_g, left_h, params)
        right_gain = calculate_gain(right_g, right_h, params)

        left_node = PredictionNode(best_node.id * 2, left_weight, left_gain, collect(left_inds), Vector{SplitterNode}(), best_y_i)
        right_node = PredictionNode(best_node.id * 2 + 1, right_weight, right_gain, collect(right_inds), Vector{SplitterNode}(), best_y_i)
        new_splitter = SplitterNode(best_split_index, best_split_value, left_node, right_node)
        push!(best_node.splitter_nodes, new_splitter)
        push!(stack, left_node)
        push!(stack, right_node)

        update_pgh!(y, left_weight, predictions, gradients, hessians, best_y_i, left_inds, params)
        update_pgh!(y, right_weight, predictions, gradients, hessians, best_y_i, right_inds, params)
    end
    return root
end

function train_minimized_sampled(x::AbstractMatrix{Int}, y::AbstractMatrix{Int}, training_params::TrainingParameters) where T <: AbstractFloat
    params = training_params.hyper_params
    bin_sizes = training_params.histogram.num_bins

    predictions, gradients, hessians = initialize_pgh(y, params)
    gradients .= params.grad_func(y, predictions) .* params.case_weights
    hessians .= params.hess_func(y, predictions) .* params.case_weights

    root_weight = 0.0
    root_gain = 0.0
    root_y_i = 0
    for y_i in 1:params.num_classes
        grad_sum = vec_sum(gradients, y_i, collect(1:size(x, 1)))
        hess_sum = vec_sum(hessians, y_i, collect(1:size(x, 1)))
        weight = calculate_weight(grad_sum, hess_sum, params)
        gain = calculate_gain(grad_sum, hess_sum, params)
        if gain > root_gain
            root_weight = weight
            root_gain = gain
            root_y_i = y_i
        end
    end
    root = PredictionNode(1, root_weight, root_gain, collect(1:size(x, 1)), Vector{SplitterNode}(), root_y_i)
    predictions[:, root_y_i] .+= root_weight .* params.learning_rate
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
        best_y_i = 0
        for i in 1:size(node_indices, 1)
            node = stack[node_indices[i]]
            for y_i in 1:params.num_classes
                gain, split_index, split_value = bin_split(x, gradients, hessians, y_i, node.indices, bin_sizes, params)
                if gain > best_gain
                    best_node_index = i
                    best_gain = gain
                    best_split_index = split_index
                    best_split_value = split_value
                    best_y_i = y_i
                end
            end
        end
        best_node = stack[node_indices[best_node_index]]
        left_inds, right_inds = split_dataset(x, best_split_index, best_split_value, best_node.indices)

        left_g, left_h = vec_sum(gradients, best_y_i, left_inds), vec_sum(hessians, best_y_i, left_inds)
        right_g, right_h = vec_sum(gradients, best_y_i, right_inds), vec_sum(hessians, best_y_i, right_inds)

        left_weight = calculate_weight(left_g, left_h, params)
        right_weight = calculate_weight(right_g, right_h, params)
        left_gain = calculate_gain(left_g, left_h, params)
        right_gain = calculate_gain(right_g, right_h, params)

        left_node = PredictionNode(best_node.id * 2, left_weight, left_gain, collect(left_inds), Vector{SplitterNode}(), best_y_i)
        right_node = PredictionNode(best_node.id * 2 + 1, right_weight, right_gain, collect(right_inds), Vector{SplitterNode}(), best_y_i)
        new_splitter = SplitterNode(best_split_index, best_split_value, left_node, right_node)
        push!(best_node.splitter_nodes, new_splitter)
        push!(stack, left_node)
        push!(stack, right_node)

        update_pgh!(y, left_weight, predictions, gradients, hessians, best_y_i, left_inds, params)
        update_pgh!(y, right_weight, predictions, gradients, hessians, best_y_i, right_inds, params)
    end
    return root
end

function train(x::AbstractMatrix{T}, y::AbstractVector{Int}, params::HyperParameters; method::Symbol=:greedy) where T <: AbstractFloat
    x_binned, hist_info = histogram(x, params)
    training_params = TrainingParameters(params, hist_info)
    y_hot = one_hot(y, params.num_classes)
    if method == :greedy
        roots = train_greedy(x_binned, y_hot, training_params)
        return MultiTree(roots, training_params)
    elseif method == :sampled
        roots = train_sampled(x_binned, y_hot, training_params)
        return MultiTree(roots, training_params)
    elseif method == :greedyforest
        roots = train_greedy_forest(x_binned, y_hot, training_params)
        return MultiTree(roots, training_params)
    elseif method == :minimizedgreedy
        root = train_minimized_greedy(x_binned, y_hot, training_params)
        return Tree(root, training_params)
    elseif method == :minimizedsampled
        root = train_minimized_sampled(x_binned, y_hot, training_params)
        return Tree(root, training_params)
    else
        error("Unknown method")
    end
end

function predict(multi_tree::MultiTree, x::AbstractMatrix{T}) where T <: AbstractFloat
    predictions = fill(multi_tree.training_params.hyper_params.initial_prediction, size(x, 1), multi_tree.training_params.hyper_params.num_classes)
    x_binned, hist_info = histogram(x, multi_tree.training_params.histogram)
    
    for y_i in 1:size(predictions, 2)
        curr_root = multi_tree.roots[y_i]
        for i in 1:size(x, 1)
            nodes = [curr_root]
            for curr_node in nodes
                predictions[i, y_i] += curr_node.weight * multi_tree.training_params.hyper_params.learning_rate
                for splitter in curr_node.splitter_nodes
                    if x_binned[i, splitter.split_feature] <= splitter.split_value
                        push!(nodes, splitter.left_node)
                    else
                        push!(nodes, splitter.right_node)
                    end
                end
            end
        end
    end
    return predictions
end

function predict(tree::Tree, x::AbstractMatrix{T}) where T <: AbstractFloat
    predictions = fill(tree.training_params.hyper_params.initial_prediction, size(x, 1), tree.training_params.hyper_params.num_classes)
    x_binned, hist_info = histogram(x, tree.training_params.histogram)

    for i in 1:size(x, 1)
        nodes = [tree.root]
        for curr_node in nodes
            y_i = curr_node.class_index
            predictions[i, y_i] += curr_node.weight * tree.training_params.hyper_params.learning_rate
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

end
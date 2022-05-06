module TreeStructs

using ..HistogramStructs
using ..TreeParameters

abstract type AbstractPredictionNode end
abstract type AbstractSplitterNode end

export AbstractPredictionNode, SplitterNode, PredictionNode, Tree, MultiTree

struct SplitterNode <: AbstractSplitterNode
    split_feature::Int
    split_value::Int
    left_node::AbstractPredictionNode
    right_node::AbstractPredictionNode
end

struct PredictionNode{T<:AbstractFloat} <: AbstractPredictionNode
    id::Int
    weight::T
    gain::T
    indices::Vector{Int}
    splitter_nodes::Vector{SplitterNode}
    class_index::Int
end

struct Tree{T<:AbstractFloat}
    root::PredictionNode{T}
    training_params::TrainingParameters{T}
end

struct MultiTree{T<:AbstractFloat}
    roots::Vector{AbstractPredictionNode}
    training_params::TrainingParameters{T}
end

end
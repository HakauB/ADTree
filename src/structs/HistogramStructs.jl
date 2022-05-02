module HistogramStructs

export HistogramInfo

struct HistogramInfo{T<:AbstractFloat}
    num_bins::Vector{Int}
    hist_limits::Vector{Vector{T}}
end
    
end
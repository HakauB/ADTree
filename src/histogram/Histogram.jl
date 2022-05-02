module Histogram

using ..TreeParameters
using ..HistogramStructs

export histogram, histogram, binindex_column

function frequency_histogram_feature(x_f::AbstractVector{T}, num_bins::Int) where T <: Number
    dv = sort(x_f)
    uniques = unique(dv)
    if num_bins > size(uniques, 1)
        bin_range = uniques
        return sort(bin_range), size(uniques, 1)
    end
    elements = size(dv, 1)
    bin_width = elements / num_bins

    bin_range = zeros(T, num_bins)
    for i in 1:num_bins
        bin_range[i] = dv[Int(floor(i * bin_width))]
    end
    bin_range[end] = dv[end]
    return sort(bin_range), num_bins
end

function find_bin(x::T, limits::Vector{T}) where T <: AbstractFloat
    bin = 0
    for i in 1:length(limits)
        if x <= limits[i]
            bin = i
            break
        end
    end
    if bin == 0
        bin = length(limits)
    end
    return bin
end

function binindex(dv::Vector{T}, limits::Vector{T}) where T <: AbstractFloat
    vbins = zeros(Int16, length(dv))
    for i in 1:length(dv)
        vbins[i] = find_bin(dv[i], limits)
    end
    return vbins
end

function binindex_column(dv::Vector{T}, limits::Vector{Vector{T}}) where T <: AbstractFloat
    vbins = zeros(Int16, length(dv))
    for i in 1:length(dv)
        vbins[i] = find_bin(dv[i], limits[i])
    end
    return vbins
end

function frequency_histogram_dataset(x::AbstractMatrix{T}, num_bins::Int) where T <: AbstractFloat
    hist_bins = Matrix{Int}(undef, size(x, 1), size(x, 2))
    bin_limits = Vector{Vector{T}}(undef, size(x, 1))
    bin_sizes = Vector{Int}(undef, size(x, 1))
    for i = 1:size(x, 1)
        bin_limits[i], bin_sizes[i] = frequency_histogram_feature(x[i, :], num_bins)
        hist_bins[i, :] = binindex(x[i, :], bin_limits[i])
    end
    return hist_bins, bin_limits, bin_sizes
end

function frequency_histogram_dataset(x::AbstractMatrix{T}, histogram::HistogramInfo{T}) where T <: AbstractFloat
    hist_bins = Matrix{Int}(undef, size(x, 1), size(x, 2))
    bin_sizes = histogram.num_bins
    bin_limits = histogram.hist_limits
    for i = 1:size(x, 1)
        hist_bins[i, :] = binindex(x[i, :], bin_limits[i])
    end
    return hist_bins, bin_limits, bin_sizes
end

function histogram(x::AbstractMatrix{T}, params::HyperParameters{T}) where T <: AbstractFloat
    if params.histogram_version == :frequency
        hist_bins, bin_limits, bin_sizes = frequency_histogram_dataset(x, params.num_bins)
        return hist_bins, HistogramInfo{T}(bin_sizes, bin_limits)
    else
        error("Unknown histogram version")
    end
end

function histogram(x::AbstractMatrix{T}, histogram_info::HistogramInfo{T}) where T <: AbstractFloat
    hist_bins, bin_limits, bin_sizes = frequency_histogram_dataset(x, histogram_info)
    return hist_bins, HistogramInfo{T}(bin_sizes, bin_limits)
end

end
# ADTree Documentation
```@meta
using .ADTree
CurrentModule = ADTree
```

## Hyper-Parameters
```@docs
train(x::AbstractMatrix{T}, y::AbstractVector{T}, params::HyperParameters) where T <: AbstractFloatstruct HyperParameters
```

## Regression Tree
### Training
```@docs
train(x::AbstractMatrix{T}, y::AbstractVector{T}, params::HyperParameters) where T <: AbstractFloat
```

### Prediction
```@docs
predict(tree::Tree, x::AbstractMatrix{T}) where T <: AbstractFloat
```

## Classification Tree
### Training
```@docs
train(x::AbstractMatrix{T}, y::AbstractVector{T}, params::HyperParameters) where T <: AbstractFloat
```

### Prediction
For non-minimized trees:
```@docs
"""
predict(multi_tree::MultiTree, x::AbstractMatrix{T}) where T <: AbstractFloat
"""

or, for minimized trees:
```@docs
predict(tree::Tree, x::AbstractMatrix{T}) where T <: AbstractFloat
```
include.(["structs/HistogramStructs.jl", "structs/TreeParameters.jl", "structs/TreeStructs.jl", "histogram/Histogram.jl", "utils/TreeUtils.jl", "tree/RegressionTree.jl"]);
using .TreeParameters
#using .Histogram
using .RegressionTree

hyper_params = HyperParameters(learning_rate=0.3, num_bins = 100)
#println(hyper_params)

using Random
Random.seed!(3)
x = rand(10000, 100)
y = vec(rand(10000))
#a, b = histogram(x, hyper_params)
#println(a)
#println(b)
x = transpose(x)
tree = train(x, y, hyper_params)
@time train(x, y, hyper_params)
tree = train(x, y, hyper_params; method=:sampled)
@time train(x, y, hyper_params; method=:sampled)
#tree = train(x, y, hyper_params; method=:greedyforest)
#@time train(x, y, hyper_params; method=:greedyforest)
#tree = train(x, y, hyper_params; method=:random)
#@time train(x, y, hyper_params; method=:random)
#
using Profile
@profile train(x, y, hyper_params; method=:sampled)
Profile.print()

#for i in 1:10
#    println(predict(tree, x[:, i]), " v ", y[i])
#end
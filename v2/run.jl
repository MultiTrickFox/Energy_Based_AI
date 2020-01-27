include("rbm.jl")



rbm = RBM(3,4)

inp = randn(1,3)


for _ in 1:10

    # rbm(inp)
    #
    # rbm()

    # @show rbm.visibles
    # @show rbm.hiddens

    #@show rbm.weights

    grads = alternating_gibbs_grads!(rbm, inp, n=10)
    # @show grads[1]
    update_weights!(rbm, grads, .1)

    #@show rbm.weights


end


println(" ")
println(" ")

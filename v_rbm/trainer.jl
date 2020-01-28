include("rbm.jl")

include("data.jl")


using Random: shuffle

using Knet: norm

using Plots: plot, plot!



rbm = RBM(out_size,20)


lr = .001

hm_epochs = 100

batch_size = 100



# inp = randn(1,3)
#
# data = [randn(1,3) for _ in 1:20]
#
# test_data = [randn(1,3) for _ in 1:20]



grad_norms = []
grad_sums = []

test_grad_sums = []


for ep in 1:hm_epochs


    # rbm(inp)
    #
    # rbm()

    # @show rbm.visibles
    # @show rbm.hiddens


    # #@show rbm.weights
    #
    # grads = alternating_gibbs_grads!(rbm, inp)
    # # @show grads[1]
    # update_weights!(rbm, grads, .1)
    #
    # #@show rbm.weights


    total_grads = nothing


    for batch in batchify(shuffle(data_train),batch_size)


        grads = batch_grads(rbm, batch)

        update_weights!(rbm, grads, lr)


        total_grads == nothing ? total_grads = grads : total_grads += grads


    end

    println("Epoch: $ep")
    @show norm(total_grads)


    push!(grad_norms, norm(total_grads))
    push!(grad_sums, sum(abs.(total_grads)))
    push!(test_grad_sums, sum(abs.(batch_grads(rbm, data_dev))))


end



p1 = plot(1:hm_epochs, grad_norms, title="grad_norms")
p2 = plot(1:hm_epochs, grad_sums, title="grad_sums")
p3 = plot(1:hm_epochs, test_grad_sums, title="dev_grad_sums")

display(plot(p1,p2,p3,layout=(3,1)))



println(" ")
println(" ")

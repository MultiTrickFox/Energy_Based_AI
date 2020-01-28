include("rbm.jl")

include("data.jl")


using Random: shuffle, shuffle!

using Knet: norm

using Plots: plot, plot!


train(;hidden_size = 2,
       lr          = .001,
       hm_epochs   = 100,
       batch_size  = 200,
      ) =
begin


    rbm = RBM(out_size,hidden_size)


    grad_norms = []
    grad_sums = []

    test_grad_sums = []
    test_grad_norms = []


    for ep in 1:hm_epochs

        total_grads = nothing


        for batch in batchify(shuffle(data_train),batch_size)


            grads = batch_grads(rbm, batch)

            update_weights!(rbm, grads, lr)


            total_grads == nothing ? total_grads = grads : total_grads += grads


        end



        println("Epoch: $ep grad norm: $(norm(total_grads))")



        push!(grad_norms, norm(total_grads))
        push!(grad_sums, sum(abs.(total_grads)))

        dev_grads = batch_grads(rbm, data_dev)

        push!(test_grad_sums, sum(abs.(dev_grads)))
        push!(test_grad_norms, norm(dev_grads))

    end

    p1 = plot(1:hm_epochs, grad_norms,      title="train_norms")
    p2 = plot(1:hm_epochs, grad_sums,       title="train_sums")
    p3 = plot(1:hm_epochs, test_grad_norms, title="dev_norms")
    p4 = plot(1:hm_epochs, test_grad_sums,  title="dev_sums")

    display(plot(p1,p2,p3,p4,layout=(4,1)))


    return rbm, [grad_norms,grad_sums,test_grad_norms,test_grad_sums]


end

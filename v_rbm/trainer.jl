include("rbm.jl")

include("data.jl")


using Knet: norm

using Plots: plot, plot!


train(;hidden_size = 8,
       lr          = .01,
       hm_epochs   = 50,
       batch_size  = 200,
      ) =
begin


    rbm = RBM(in_size,hidden_size)


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


        push!(grad_norms, norm(total_grads))
        push!(grad_sums, sum(abs.(total_grads)))

        dev_grads = batch_grads(rbm, data_dev)

        push!(test_grad_sums, sum(abs.(dev_grads)))
        push!(test_grad_norms, norm(dev_grads))


        println("Epoch $ep; train_norm: $(round(norm(total_grads),digits=3)), dev_norm: $(round(norm(dev_grads),digits=3))")


    end

    min_train_norm = argmin(grad_norms)
    min_train_sum = argmin(grad_sums)
    min_dev_norm = argmin(test_grad_norms)
    min_dev_sum = argmin(test_grad_sums)


    p1 = plot(1:hm_epochs, grad_norms,     title="train_norm_$(lr)_$(hidden_size)_$(batch_size)",xlabel="$(grad_norms[min_train_norm]) / $(min_train_norm)")
    p2 = plot(1:hm_epochs, grad_sums,      title="train_sum_$(lr)_$(hidden_size)_$(batch_size)",xlabel="$(grad_sums[min_train_sum]) / $(min_train_sum)")
    p3 = plot(1:hm_epochs, test_grad_norms,title="dev_norm_$(lr)_$(hidden_size)_$(batch_size)",xlabel="$(test_grad_norms[min_dev_norm]) / $(min_dev_norm)")
    p4 = plot(1:hm_epochs, test_grad_sums, title="dev_sum_$(lr)_$(hidden_size)_$(batch_size)",xlabel="$(test_grad_sums[min_dev_sum]) / $(min_dev_sum)")

    display(plot(p1,p2,p3,p4,layout=(2,2)))


rbm, [grad_norms,grad_sums,test_grad_norms,test_grad_sums]
end

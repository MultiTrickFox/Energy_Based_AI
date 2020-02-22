include("rbm.jl")

include("data.jl")


using Knet: norm

using Plots: plot, plot!


train(;rbm            = nothing,
       hidden_size    = 10,
       learning_rate  = 1,
       hm_epochs      = 1,
       batch_size     = length(data_train),
       do_print       = true
      ) =
begin


    rbm == nothing ?
        rbm = RBM(in_size,hidden_size) :
            hidden_size = length(rbm.hiddens)


    do_print ? (@info "Training started. \nhidden size   $(hidden_size) \nlearning rate $(learning_rate) \nbatch size    $(batch_size)") : ()


    grad_norms      = []
    grad_sums       = []

    test_grad_norms = []
    test_grad_sums  = []


    do_print ? (begin

        dev_grads = batch_grads(rbm, data_dev)
        println("initial dev norm $(norm(dev_grads))")
        println("initial dev sum $(sum(abs.(dev_grads)))")

    end) : ()


    for ep in 1:hm_epochs

        total_grads = nothing

        for batch in batchify(shuffle(data_train),batch_size)

            grads = batch_grads(rbm, batch)

            update_weights!(rbm, grads, learning_rate)

            total_grads == nothing ? total_grads = grads : total_grads += grads

        end

        dev_grads = batch_grads(rbm, data_dev)


        push!(grad_norms, norm(total_grads)./sqrt(length(rbm.weights)))
        push!(grad_sums, sum(abs.(total_grads))./length(rbm.weights))
        push!(test_grad_norms, norm(dev_grads)./sqrt(length(rbm.weights)))
        push!(test_grad_sums, sum(abs.(dev_grads))./length(rbm.weights))


        do_print ? println("Epoch $ep, train_sum $(round(grad_sums[end],digits=3)), dev_sum $(round(test_grad_sums[end],digits=3))") : ()


    end


    min_train_norm = argmin(grad_norms)
    min_train_sum = argmin(grad_sums)
    min_dev_norm = argmin(test_grad_norms)
    min_dev_sum = argmin(test_grad_sums)

    p1 = plot(1:hm_epochs, grad_norms,     title="train_norm_$(hidden_size)_$(learning_rate)_$(batch_size)",xlabel="$(grad_norms[min_train_norm]) / $(min_train_norm)")
    p2 = plot(1:hm_epochs, grad_sums,      title="train_sum_$(hidden_size)_$(learning_rate)_$(batch_size)",xlabel="$(grad_sums[min_train_sum]) / $(min_train_sum)")
    p3 = plot(1:hm_epochs, test_grad_norms,title="dev_norm_$(hidden_size)_$(learning_rate)_$(batch_size)",xlabel="$(test_grad_norms[min_dev_norm]) / $(min_dev_norm)")
    p4 = plot(1:hm_epochs, test_grad_sums, title="dev_sum_$(hidden_size)_$(learning_rate)_$(batch_size)",xlabel="$(test_grad_sums[min_dev_sum]) / $(min_dev_sum)")

    display(plot(p1,p2,p3,p4,layout=(2,2)))


rbm, [grad_norms,grad_sums,test_grad_norms,test_grad_sums]
end


##


using Knet: relu

using Images: Gray


generate(rbm;hidden_state=nothing,converge=true) =
begin

    if hidden_state == nothing

        random_states = randn(1,length(rbm.hiddens))

        binary ? random_states = binarize_state.(random_states) : ()

        hidden_state = random_states

    end

    rbm.hiddens = hidden_state

    rbm()

    converge ? propogate_until_convergence!(rbm, rbm.visibles) : ()

    Gray.(reshape(relu.(rbm.visibles), (int(sqrt(in_size)),int(sqrt(in_size))))')

end


##

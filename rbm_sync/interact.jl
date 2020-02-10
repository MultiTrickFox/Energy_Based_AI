include("rbm.jl")

include("data.jl")


using Knet: norm

using Plots: plot, plot!


train(;rbm            = nothing,
       hidden_size    = 10,
       learning_rate  = 1,
       hm_epochs      = 10,
       batch_size     = length(data_train),
      ) =
begin


    @info "Starting training... \nhidden size   $(hidden_size) \nlearning rate $(learning_rate) \nbatch size    $(batch_size)"


    rbm == nothing ?
        rbm = RBM(in_size,hidden_size) :
            ()


    grad_norms      = []
    grad_sums       = []

    test_grad_norms = []
    test_grad_sums  = []


    dev_grads = batch_grads(rbm, data_dev)

    println("initial dev norm $(norm(dev_grads))")
    println("initial dev sum $(sum(abs.(dev_grads)))")


    for ep in 1:hm_epochs

        total_grads = nothing

        for batch in batchify(shuffle(data_train),batch_size)


            grads = batch_grads(rbm, batch)

            update_weights!(rbm, grads, learning_rate)


            total_grads == nothing ? total_grads = grads : total_grads += grads


        end


        push!(grad_norms, norm(total_grads))
        push!(grad_sums, sum(abs.(total_grads)))

        dev_grads = batch_grads(rbm, data_dev)

        push!(test_grad_sums, sum(abs.(dev_grads)))
        push!(test_grad_norms, norm(dev_grads))


        println("Epoch $ep, train_norm $(round(norm(total_grads),digits=3)), dev_norm $(round(norm(dev_grads),digits=3))")


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


generate(rbm) =
begin


    random_states = randn(1,length(rbm.hiddens))

    binary ? random_states = binarize.(random_states) : ()


    rbm.hiddens = random_states # TODO : start from random or similar to a datapoint ?

    rbm()


reshape(rbm.visibles, (1, int(sqrt(in_size)),int(sqrt(in_size))))
end



# model, meta = train()
#
# gen = generate(model)
#
# println(" ") # show as img.



main() = begin

    for i in 1:10

        @info "Global Iteration g$(i)"

        for hs in [2,4,8,10,16,32]

            #rbm = RBM(in_size, hs)

            for lr in [1,.8,.6,.4,.2,.1]

                train(hidden_size=hs,learning_rate=lr)#,rbm=deepcopy(rbm))

            end

        end

    end

end; main()
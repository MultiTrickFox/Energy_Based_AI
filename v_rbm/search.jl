include("trainer.jl")


# using Plotly: plot
# using Plots: pyplot


hidden_sizes = [2^i for i in 2:8]

learning_rates = [1/(10^i) for i in 1:7]

hm_epochs = 100



exec() = begin


grid_results = [[[] for _ in 1:length(learning_rates)] for _ in 1:length(hidden_sizes)]

prev_norm = 999_999_999


    for (hs, res1) in zip(hidden_sizes, grid_results)

        println("\ths: $hs")


        # model = RBM(in_size, hs)
        #
        # model_grad = batch_grads(model, data_train)
        #
        #
        # train_norm = norm(model_grad)
        #
        # @show train_norm


        for (lr, (i, res2)) in zip(learning_rates, enumerate(res1))

            println("\t\tlr: $lr")


            # model_cpy = deepcopy(model)
            #
            # update_weights!(model_cpy, model_grad, lr)
            #
            #
            # dev_grad = batch_grads(model_cpy, data_dev)
            #
            # meta = [[norm(model_grad)],[sum(abs.(model_grad))],[norm(dev_grad)],[sum(abs.(dev_grad))]]
            #
            #
            # dev_norm = meta[3][end]
            #
            # @show dev_norm


            _,meta = train(hidden_size=hs,lr=lr,batch_size=len(data_train),hm_epochs=hm_epochs)


            min_dev_norm = "$(round(meta[3][argmin(meta[3])],digits=3))/$(argmin(meta[3]))"
            @show min_dev_norm
            min_dev_norm = round(meta[3][argmin(meta[3])],digits=3)

            res1[i] = meta


            min_dev_norm > prev_norm ? break : prev_norm = min_dev_norm


        end

        prev_norm = 999_999_999

    end
    # @show plot_results
    #
    # plot_results = [[e2[1][3] for e2 in e1] for e1 in grid_results] # [train_norms,train_sums,dev_norms,dev_sums]

    # plot_results = hcat(plot_results...)
    #
    # plt3d = Plots.plot(learning_rates,hidden_sizes,plot_results,seriestype=:scatter,markersize=7)
    # display(plt3d)


    # pyplot()
    #
    # x = learning_rates
    # y = hidden_sizes
    #
    # z(x,y) = plot_results[indexof(x, learning_rates)][indexof(y, learning_rates)]
    #
    # plot(x,y,z,st=:surface,camera=(-30,30))


    # for (lr, res1) in zip(learning_rates, grid_results)
    #
    #     for (hs, res2) in zip(hidden_sizes, res1)


end
 ; exec()

include("trainer.jl")


# using Plotly: plot
# using Plots: pyplot


hidden_sizes = [2^i for i in 2:10]

learning_rates = [1/(10^(i-1)) for i in 1:12]

hm_epochs = 50



exec() = begin


grid_results = [[[] for _ in 1:length(learning_rates)] for _ in 1:length(hidden_sizes)]


    for (hs, res1) in zip(hidden_sizes, grid_results)

        print("\ths: $hs")


        # model = RBM(in_size, hs)
        #
        # model_grad = batch_grads(model, data_train)
        #
        #
        # train_norm = norm(model_grad)
        #
        # @show train_norm


        for (lr, (i, res2)) in zip(learning_rates, enumerate(res1))

            print("\t\tlr: $lr")


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


            _,meta = train(hidden_size=hs,lr=lr,batch_size=len(data_train),hm_epochs=50)


            devnorm = "$(round(meta[3][argmin(meta[3])],digits=3))/$(argmin(meta[3]))"
            @show devnorm

            res1[i] = [e[end] for e in meta]

        end

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

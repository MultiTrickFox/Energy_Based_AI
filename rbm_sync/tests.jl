# include("rbm.jl")

# include("data.jl")

include("interact.jl")


##


model = nothing

main() = begin

    global model ; model = RBM(in_size, 10)

    batch = [data_train[1], data_train[2]]

    update_weights!(model, batch_grads(rbm, batch), 1)

    generate(model)

end


##


test_basic() = begin


    rbm = RBM(10, 12)

    datapoint = binarize_data.(rand(1,10))


    inp = datapoint

    prev_visibles = nothing
    prev_hiddens = nothing

    for i in 1:5

        rbm(inp)
        rbm()

        inp = rbm.visibles

        println("iteration $(i)")
        @show rbm.visibles
        @show rbm.hiddens
        @show energy(rbm)

        prev_visibles == rbm.visibles ? println("visibles stable.") : prev_visibles = rbm.visibles
        prev_hiddens == rbm.hiddens ? println("hiddens stable.") : prev_hiddens = rbm.hiddens

    end


end ; #test_basic()


##


test_bestsize() = begin

    hm_avg = 100

    hidden_sizes = 1:12

    results = [0.0 for _ in hidden_sizes]

    for ii in 1:hm_avg

        for (i,hs) in enum(hidden_sizes)

            results[i] = (results[i] * (ii-1) + train(hidden_size=hs,hm_epochs=1,do_print=false)[end][end][end]) / ii

        end

        println("iteration: $(ii), results:")
        for (hs, res) in zip(hidden_sizes, results)
            println("\thidden_size: $(hs), loss: $(res)")
        end
        println("minimum loss: $(hidden_sizes[argmin(results)]) - $(argmin(results))")

    end

    println("Final Results:")
    for (hs,res) in zip(hidden_sizes, results)
        println("\t$(hs): $(res)")
    end

end ; #test_bestsize()


##


; main()

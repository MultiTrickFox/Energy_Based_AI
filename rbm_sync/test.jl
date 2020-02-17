# include("rbm.jl")

# include("data.jl")

include("interact.jl")


data_train = [e == 0 ? -1 : e for e in data_train]
data_dev = [e == 0 ? -1 : e for e in data_dev]


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


test_hiddensizes() = begin

    hm_avg = 100

    hidden_sizes = 1:12

    results = [0.0 for _ in hidden_sizes]

    for ii in 1:hm_avg

        for (i,hs) in enum(hidden_sizes)

            results[i] = (results[i] * (ii-1) + train(hidden_size=hs,hm_epochs=1)[end][end][end]) / ii

        end

        println("iteration: $(ii), lowest loss: $(hidden_sizes[argmin(results)])")

    end

    # results = [e/hm_avg for e in results]

    println("Final Results:")

    for (hs,res) in zip(hidden_sizes, results)

        println("\t$(hs): $(res)")

    end

end ; #test_hiddensizes()


##


# main() = begin
#
#     for i in 1:10
#
#         @info "Global Iteration g$(i)"
#
#         for hs in [2,4,8,10,16,32]
#
#             #rbm = RBM(in_size, hs)
#
#             for lr in [1,.8,.6,.4,.2,.1]
#
#                 train(hidden_size=hs,learning_rate=lr)#,rbm=deepcopy(rbm))
#
#             end
#
#         end
#
#     end
#
# end; main()

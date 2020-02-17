# include("rbm.jl")

# include("data.jl")

include("interact.jl")


##


test_basic() = begin


    rbm = RBM(10, 12)

    datapoint = binarize_data.(rand(1,10)) # data_train[1]


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


test_train() = begin


    rbm = RBM(in_size,10)

    random_state = randn(1,length(rbm.hiddens))

    binary ? random_state = binarize_state.(random_state) : ()

    gen1 = generate(rbm,hidden_state=random_state)

    rbm, meta = train(rbm=rbm,hm_epochs=1)

    gen2 = generate(rbm,hidden_state=random_state)


    println(" ")

    gen1 == gen2 ? (println("same.")) : (println("smt changed."))
    ctr = 0
    for (e1,e2) in zip(gen1,gen2)
        e1 != e2 ? ctr +=1 : ()
    end
    println("diff ratio: $(ctr/in_size)")


    # using ImageView: imshow
    #
    # imshow(gen2)


end ; #test_train()


test_hiddensizes() = begin

    hm_avg = 100

    hidden_sizes = 1:12

    results = [0 for _ in hidden_sizes]

    for _ in 1:hm_avg

        for (i,hs) in enum(hidden_sizes)

            results[i] += train(hidden_size=hs,hm_epochs=1)[end][end][end]

        end

    end

    results = [e/hm_avg for e in results]

    for (hs,res) in zip(hidden_sizes, results)

        println("$(hs): $(res)")

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

include("utils.jl")


binary = false


binarize_data(x) = round(x) <= 0 ? -1 : 1

binarize_state(x) = x <= 0 ? -1 : 1


##


mutable struct RBM

    visibles::Array{Float32}
    hiddens::Array{Float32}

    weights::Array{Float32}

    RBM(in_size, hidden_size) = new(

        [0 for _ in 1:in_size],
        [0 for _ in 1:hidden_size],

        randn(in_size, hidden_size),

    )

end


(rbm::RBM)(input) =
begin

    rbm.hiddens = tanh.(input * rbm.weights)

    binary ? rbm.hiddens = binarize_state.(rbm.hiddens) : ()

end


(rbm::RBM)() =
begin

    rbm.visibles = tanh.(rbm.hiddens * transpose(rbm.weights))

    binary ? rbm.visibles = binarize_state.(rbm.visibles) : ()

end



update_weights!(rbm, grads, lr) =
begin

    rbm.weights += lr .* grads

end



alternating_gibbs_grads!(rbm, input; k=1) =
begin

    rbm(input)

    pos_hiddens = rbm.hiddens

    for _ in 1:k

        rbm()

        rbm(rbm.visibles)

    end

    grads = (transpose(input) * pos_hiddens) .- (transpose(rbm.visibles) * rbm.hiddens)

grads
end


batch_grads(rbm, batch) = threadpool(sum, args->alternating_gibbs_grads!(args...), [[deepcopy(rbm), input] for input in batch]) ./ length(batch)


##


energy(rbm) = -(rbm.visibles * rbm.weights * rbm.hiddens')



propogate_until_convergence!(rbm, input; k=1_000) =
begin

    prev_visibles = nothing
    prev_hiddens = nothing

    ctr = 0

    while (rbm.visibles != prev_visibles || rbm.hiddens != prev_hiddens) && ctr < k

        prev_visibles = rbm.visibles
        prev_hiddens = rbm.hiddens

        rbm(input)
        rbm()

        input = rbm.visibles

        ctr +=1

    end

input
end

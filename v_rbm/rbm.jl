include("utils.jl")

using Knet: sigmoid



binary = false

non_binary_act = sigmoid   # TODO : ask, tanh, sigm, & continuous


binarize(x) = round(x)

    # x > 0 ? 1 : 0     # TODO : ask, which one to use


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

    if binary
        rbm.hiddens = binarize.(sigmoid.(input * rbm.weights))
    else
        rbm.hiddens = non_binary_act.(input * rbm.weights)
    end


(rbm::RBM)() =

    if binary
        rbm.visibles = binarize.(sigmoid.(rbm.hiddens * transpose(rbm.weights)))
    else
        rbm.visibles = non_binary_act.(rbm.hiddens * transpose(rbm.weights))
    end



update_weights!(rbm, grad, learning_rate) =
begin

    rbm.weights += learning_rate .* grad

end


propogate!(rbm, input) =
begin

    rbm(input)
    rbm()

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

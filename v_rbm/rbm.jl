include("utils.jl")

# using Knet: sigmoid



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

end

(rbm::RBM)() =
begin

    rbm.visibles = tanh.(rbm.hiddens * transpose(rbm.weights))

end


update_weights!(rbm, learning_rate) =
begin

    rbm.weights += learning_rate .* (transpose(rbm.visibles) * rbm.hiddens)

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

    current_visibles = input
    pos_hiddens = nothing

    for _ in 1:k

        propogate!(rbm, current_visibles)

        current_visibles = rbm.visibles

        pos_hiddens == nothing ? pos_hiddens = rbm.hiddens : ()

    end

    grads = (transpose(input) * pos_hiddens) .- (transpose(rbm.visibles) * rbm.hiddens)

grads
end


batch_grads(rbm, batch) = threadpool(sum, args->alternating_gibbs_grads!(args...), [[deepcopy(rbm), input] for input in batch]) ./ length(batch)

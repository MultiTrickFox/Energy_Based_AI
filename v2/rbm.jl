include("utils.jl")

# using Knet: KnetArray # unlock dis.
using Knet: sigmoid



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

    rbm.hiddens = sigmoid.(input * rbm.weights)

end

(rbm::RBM)() =
begin

    rbm.visibles = sigmoid.(rbm.hiddens * transpose(rbm.weights))

end


update_weights!(rbm, learning_rate) =
begin

    rbm.weights += learning_rate .* (transpose(rbm.visibles) * rbm.hiddens)

end

update_weights!(rbm, grad, learning_rate) =
begin

    rbm.weights += learning_rate .* grad

end


propogate!(rbm, input; n=1) =

    for _ in 1:n

        rbm(input)
        rbm()

    end


alternating_gibbs_grads!(rbm, input; n=1) =
begin

    current_visibles = input
    pos_hiddens = nothing

    for _ in 1:n

        propogate!(rbm, current_visibles)

        current_visibles = rbm.visibles

        pos_hiddens == nothing ? pos_hiddens = rbm.hiddens : ()

    end ; @show current_visibles ; @show input

    grads = (transpose(input) * pos_hiddens) .- (transpose(rbm.visibles) * rbm.hiddens)

grads
end

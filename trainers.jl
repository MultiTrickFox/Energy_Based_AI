using Distributed: @distributed, addprocs, procs
length(procs()) == 1 ? addprocs(Sys.CPU_THREADS) : ()

include("models.jl")


# Hopfield


grads_on_data(hopfield::Hopfield, input; thermal=false) =
begin

    thermal ? binary_update_thermal(hopfield, input)
        binary_update(hopfield, input)

    grads = hebbian_grads(hopfield)

    for node in hopfield.nodes
        node.state = 0
    end

grads
end

grads_on_batch(hopfield::Hopfield, batch; thermal=false) =

    (@distributed (sum) for data in batch
        grads_on_data(hopfield, data, thermal=thermal)
    end) / length(batch)


# Binary Boltzmann


grads_on_data(boltzmann::Boltzmann, input; binary=true) =
begin

    binary ? binary_update_thermal_hiddens(boltzmann, input) :
        continuous_update_thermal_hiddens(boltzmann, input)

    grads = hebbian_grads(boltzmann)

    for node in boltzmann.nodes
        node.state = 0
    end

grads
end

# grads_on_data_negpos(boltzmann::Boltzmann, input, k; binary=true) =
# begin
#                            # TODO : DO
#     if binary
#         binary_update_thermal_hiddens(boltzmann, input) :
#         continuous_update_thermal_hiddens(boltzmann, input)
#
#     grads = hebbian_grads(boltzmann)
#
#     for node in boltzmann.nodes
#         node.state = 0
#     end
#
# grads
# end



grads_on_batch(hopfield::Hopfield, batch; thermal=false) =

    (@distributed (sum) for data in batch
        grads_on_data(hopfield, data)
    end) / length(batch)





# TODO : negative positive trainig



# Continuous Bolztmann






# batch = [input, input]
#
# @show grads_on_batch(hopfield, batch)

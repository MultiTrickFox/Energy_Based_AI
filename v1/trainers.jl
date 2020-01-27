using Distributed: @distributed, addprocs, procs
using Distributed: @everywhere
length(procs()) == 1 ? addprocs(Sys.CPU_THREADS) : ()

@everywhere include("models.jl")


# Hopfield


@everywhere grads_on_data(hopfield::Hopfield, input; t=1) =
begin

    binary_update_thermal!(hopfield, input, t)

hebbian_grads(hopfield)
end

grads_on_batch(hopfield::Hopfield, batch; t=1) =

    # (@distributed (sum) for data in batch
    #     grads_on_data(hopfield, data, t=t)
    # end) / length(batch)

    sum([grads_on_data(hopfield, data, t=t) for data in batch]) / length(batch)


# Boltzmann


@everywhere grads_on_data(boltzmann::Boltzmann, input; binary=false, t=1) =
begin

    binary ? binary_update_thermal_hiddens(boltzmann, input, t) :
        continuous_update_thermal_hiddens(boltzmann, input, t)

hebbian_grads(boltzmann)
end

grads_on_data_negpos(boltzmann::Boltzmann, input, k; binary=false, t=1) =
begin

    binary ? binary_update_thermal_hiddens(boltzmann, input, t) :
        continuous_update_thermal_hiddens(boltzmann, input, t)

hebbian_grads(boltzmann)
end


grads_on_batch(boltzmann::Boltzmann, batch; binary=false, t=1, negpos=true, k=1) =

    negpos ?

        (@distributed (sum) for data in batch
            grads_on_data_negpos(boltzmann, data, k, binary=binary, t=t)
        end) / length(batch) :

        (@distributed (sum) for data in batch
            grads_on_data(boltzmann, data, binary=binary, t=t)
        end) / length(batch)





# temperature_calculate(iteration, initial_t) =
    # TODO : decrease temperature wrt sin wave, triangle wave etc wrt given math func

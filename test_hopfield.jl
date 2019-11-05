include("models.jl")
include("trainers.jl")


main() =
begin


    epochs = 20
    lr = .1

    temperature_initial = 2
    temperature_decay = .5


    input1 = [-1,1,-1,1,1,-1,-1]
    input2 = [-1,-1,1,1,-1,1,1]
    batch = [input1, input2]

    hopfield = Hopfield(length(input))


    println("> binary hopfield:")

    for _ in 1:epochs

        grads = grads_on_batch(hopfield, batch)
        hebbian_learn!(hopfield, grads, lr)

    end ; println(" ")


    println("> binary thermal hopfield:")

    temperature = temperature_initial

    for _ in 1:epochs

        grads = grads_on_batch(hopfield, batch)
        hebbian_learn!(hopfield, grads, lr)
        temperature *= temperature_decay

    end ; println(" ")


end ; main()

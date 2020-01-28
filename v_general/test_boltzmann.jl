include("models.jl")
include("trainer.jl")


main() =
begin


    epochs = 20
    lr = .1

    hm_hiddens = 4

    temperature_initial = 2
    temperature_decay = .5


    input = [-1,1,-1,1,1,-1,-1]
    boltzmann = Boltzmann(length(input), hm_hiddens)


    


end ; main()

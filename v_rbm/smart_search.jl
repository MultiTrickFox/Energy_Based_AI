include("trainer.jl")



runner() = begin


    learning_rates = [.1,.01,.001,.0001]

    hidden_sizes = [2^i for i in 4:10]


    hm_epochs_max = 10



    ## grid bookkeeping ##


    results = [[[] for _ in hidden_sizes] for _ in learning_rates]


    current_lr = .01 #learning_rates[rand(1:length(learning_rates))]
    current_hs = 16 #hidden_sizes[rand(1:length(hidden_sizes))]

    prev_lr = nothing
    prev_hs = nothing

    prev_loss = 999_999_999

    prev_tried_axis = "hs"


    ##


    while true


        println("\nlearning rate: $current_lr hidden size: $current_hs")

        (model,result) = train(hidden_size=current_hs,lr=current_lr,hm_epochs=hm_epochs_max)[end]

        push!(results[indexof(current_lr,learning_rates)][indexof(current_hs,hidden_sizes)], result)


        current_loss = result[end-1][argmin(result[end-1])]

        @show current_loss


        # println(" ")
        # println(" ")



        if current_loss < prev_loss

            println("** new best found! : ($(current_lr),$(current_hs))")

            if current_lr != prev_lr
                if current_lr < prev_lr
                    next_lr = learning_rates[indexof(prev_lr,learning_rates)-1]
                else
                    next_lr = learning_rates[indexof(prev_lr,learning_rates)+1]
                end
                next_hs = current_hs

            elseif current_hs != prev_hs
                if current_hs < prev_hs
                    next_hs = hidden_sizes[indexof(prev_hs,hidden_sizes)-1]
                else
                    next_hs = hidden_sizes[indexof(prev_hs,hidden_sizes)+1]
                end
                next_lr = current_lr

            end

            prev_lr = current_lr
            prev_hs = current_hs

            current_lr = next_lr
            current_hs = next_hs

            prev_loss = current_loss

        else

            if current_lr != prev_lr

                if prev_tried_axis == "lr"
                    next_lr = prev_lr
                    next_hs = hidden_sizes[indexof(prev_hs, hidden_sizes)+1]
                else
                    if current_lr < prev_lr
                        next_lr = learning_rates[indexof(prev_lr,learning_rates)+1]
                    else
                        next_lr = learning_rates[indexof(prev_lr,learning_rates)-1]
                    end
                    next_hs = current_hs
                    prev_tried_axis = "lr"
                end

            elseif current_hs != prev_hs

                if prev_tried_axis == "hs"
                    next_lr = learning_rates[indexof(prev_lr, learning_rates)+1]
                    next_hs = prev_hs
                else
                    if current_hs < prev_hs
                        next_hs = hidden_sizes[indexof(prev_hs,hidden_sizes)+1]
                    else
                        next_hs = hidden_sizes[indexof(prev_hs,hidden_sizes)-1]
                    end
                    next_lr = current_lr
                    prev_tried_axis = "hs"
                end

            end

            current_lr = next_lr
            current_hs = next_hs

        end


    end


end ; runner()


# TODO for future : train (& select further) by reducing batch_size manually.

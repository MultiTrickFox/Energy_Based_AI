## IMPORTS


using Random: shuffle, rand
using Knet: sigm


## PARAMS


max_update_steps             = 5_000

boltzmann_full_restricted    = true
boltzmann_hm_initial_configs = 15


## BASIC STRUCTS


mutable struct Node

    state
    edges

    Node(state=0.0, edges=[]) = new(
        state,
        edges,
    )

end

mutable struct Edge

    node_from
    node_to
    weight

    Edge(node_from, node_to; weight=randn()) = new(
        node_from,
        node_to,
        weight
    )

end

mutable struct Edge_Sym

    node_from
    node_to
    weight
    edge_sym

    Edge_Sym(node_from, node_to; weight=randn(), edge_sym=nothing) = new(
        node_from,
        node_to,
        weight,
        edge_sym
    )

end


## ADV STRUCTS


mutable struct Hopfield

    nodes
    edges

    Hopfield(in_size) = begin

        nodes = [Node() for _ in 1:in_size]
        edges = []

        for i in 1:in_size
            for j in i+1:in_size
                edge1 = Edge_Sym(nodes[i], nodes[j])
                edge2 = Edge_Sym(nodes[j], nodes[i], weight=edge1.weight)
                edge1.edge_sym = edge2
                edge2.edge_sym = edge1
                push!(edges, edge1)
                push!(edges, edge2)
                push!(nodes[i].edges, edge1)
                push!(nodes[j].edges, edge2)
            end
        end

    new(nodes, edges)
    end

end

mutable struct Boltzmann

    visibles
    hiddens
    edges

    Boltzmann(in_size, hidden_size; full_restricted=boltzmann_full_restricted) = begin

        visibles = [Node() for _ in 1:in_size]
        hiddens = [Node() for _ in 1:hidden_size]
        edges = []

        for i in 1:in_size
            for j in 1:hidden_size
                edge1 = Edge_Sym(visibles[i], hiddens[j])
                edge2 = Edge_Sym(hiddens[j], visibles[i], weight=edge1.weight)
                edge1.edge_sym = edge2
                edge2.edge_sym = edge1
                push!(edges, edge1)
                push!(edges, edge2)
                push!(visibles[i].edges, edge1)
                push!(hiddens[j].edges, edge2)
            end
        end

        if !full_restricted

            for i in 1:hidden_size
                for j in i+1:hidden_size
                    edge1 = Edge_Sym(hiddens[i], hiddens[j])
                    edge2 = Edge_Sym(hiddens[j], hiddens[i], weight=edge1.weight)
                    edge1.edge_sym = edge2
                    edge2.edge_sym = edge1
                    push!(edges, edge1)
                    push!(edges, edge2)
                    push!(hiddens[i].edges, edge1)
                    push!(hiddens[j].edges, edge2)
                end
            end

        end

    new(visibles, hiddens, edges)
    end

end


## STATE UPDATERS


    # Binary Hopfield


binary_update(hopfield::Hopfield, input; update_steps=max_update_steps) =
begin

    for (node, inp) in zip(hopfield.nodes, input)
        node.state = inp
    end

    last_states, ctr = nothing, 0

    while ctr <= update_steps

        for node in shuffle(hopfield.nodes)
            node.state = sign(sum([edge.weight * edge.node_to.state for edge in node.edges]))
        end

        current_states = [node.state for node in hopfield.nodes]

        if current_states == last_states
            break
        else
            last_states = current_states
            ctr +=1
        end

    end

last_states
end

binary_update_thermal(hopfield::Hopfield, input, temperature; update_steps=max_update_steps) =
begin

    for (node, inp) in zip(hopfield.nodes, input)
        node.state = inp
    end

    last_states, ctr = nothing, 0

    while ctr <= update_steps

        for node in shuffle(hopfield.nodes)
            prob = sigm(sum([edge.weight * edge.node_to.state for edge in node.edges])/temperature)
            randn() <= prob ? node.state = 1.0 : node.state = -1.0
        end

        current_states = [node.state for node in hopfield.nodes]

        if current_states == last_states
            break
        else
            last_states = current_states
            ctr +=1
        end

    end

last_states
end


    # Binary Boltzmann


binary_update_thermal_hiddens(boltzmann::Boltzmann, input, temperature; update_steps=max_update_steps, hm_average=boltzmann_hm_initial_configs) =
begin

    for (node, inp) in zip(boltzmann.visibles, input)
        node.state = inp
    end

    final_states = [.0 for _ in 1:length(boltzmann.hiddens)]

    for _ in 1:hm_average

        for node in boltzmann.hiddens
            node.state = round(rand())
        end

        last_states, ctr = nothing, 0

        while ctr <= update_steps

            for node in shuffle(boltzmann.hiddens)
                prob = sigm(sum([edge.weight * edge.node_to.state for edge in node.edges])/temperature)
                randn() <= prob ? node.state = 1.0 : node.state = -1.0
            end

            current_states = [node.state for node in boltzmann.hiddens]

            if current_states == last_states
                break
            else
                last_states = current_states
                ctr +=1
            end

        end

        final_states += last_states

    end

    final_states /= hm_average

final_states
end

binary_update_thermal_visibles(boltzmann::Boltzmann, hiddens, temperature; update_steps=max_update_steps, hm_average=update_steps=max_update_steps) =
begin

    for (node, hidden) in zip(boltzmann.hiddens, hiddens)
        node.state = hidden
    end

    final_states = [.0 for _ in 1:length(boltzmann.visibles)]

    for _ in 1:hm_average

        for node in boltzmann.input
            node.state = round(rand())
        end

        last_states, ctr = nothing, 0

        while ctr <= update_steps

            for node in shuffle(boltzmann.visibles)
                prob = sigm(sum([edge.weight * edge.node_to.state for edge in node.edges])/temperature)
                randn() <= prob ? node.state = 1.0 : node.state = -1.0
            end

            current_states = [node.state for node in boltzmann.visibles]

            if current_states == last_states
                break
            else
                last_states = current_states
                ctr +=1
            end

        end

        final_states += last_states

    end

    final_states /= hm_average

final_states
end


    # Continuous Boltzmann


continuous_update_thermal_hiddens(boltzmann::Boltzmann, input, temperature; update_steps=max_update_steps, hm_average=update_steps=max_update_steps) =
begin

    for (node, inp) in zip(boltzmann.visibles, input)
        node.state = inp
    end

    final_states = [0 for _ in 1:length(boltzmann.hiddens)]

    for _ in 1:hm_average

        for node in boltzmann.hiddens
            node.state = randn()
        end

        last_states, ctr = nothing, 0

        while ctr <= update_steps

            for node in shuffle(boltzmann.hiddens)
                node.state = sigm(sum([edge.weight * edge.node_to.state for edge in node.edges])/temperature)
            end

            current_states = [node.state for node in boltzmann.hiddens]

            if current_states == last_states
                break
            else
                last_states = current_states
                ctr +=1
            end

        end

        final_states += last_states

    end

    final_states /= hm_average

final_states
end

continuous_update_thermal_visibles(boltzmann::Boltzmann, hiddens, temperature; update_steps=max_update_steps, hm_average=update_steps=max_update_steps) =
begin

    for (node, hidden) in zip(boltzmann.visibles, hiddens)
        node.state = hidden
    end

    final_states = [.0 for _ in 1:length(boltzmann.visibles)]

    for _ in 1:length(boltzmann.visibles)

        for node in boltzmann.visibles
            node.state = randn()
        end

        last_states, ctr = nothing, 0

        while ctr <= update_steps

            for node in shuffle(boltzmann.visibles)
                node.state = sigm(sum([edge.weight * edge.node_to.state for edge in node.edges])/temperature)
            end

            current_states = [node.state for node in boltzmann.visibles]

            if current_states == last_states
                break
            else
                last_states = current_states
                ctr +=1
            end

        end

        final_states += last_states

    end

    final_states /= hm_average

final_states
end


# HEBBIAN OPERATORS


hebbian_grads(hopfield::Hopfield) =
begin

[edge.node_from.state * edge.node_to.state for edge in hopfield.edges]
end

hebbian_grads(boltzmann::Boltzmann) =
begin

[edge.node_from.state * edge.node_to.state for edge in boltzmann.edges]
end


hebbian_learn(hopfield::Hopfield, grads, lr) =

    for (edge, grad) in zip(hopfield.edges, grads)
        edge.weight += lr * grad
    end

hebbian_learn(boltzmann::Boltzmann, grads, lr) =

    for (edge, grad) in zip(boltzmann.edges, grads)
        edge.weight += lr * grad
    end


## TESTS


input = [-1,1,-1,1,1,-1,-1]

hopfield = Hopfield(length(input))
boltzmann = Boltzmann(length(input), 4)

main() =
begin

    epochs = 20
    lr = .1

    temperature_initial = 1
    temperature_decay = .5

    println("> binary hopfield:")

    for _ in 1:epochs

        states = binary_update(hopfield, input)
        grads = hebbian_grads(hopfield)
        hebbian_learn(hopfield, grads, lr)

        @show states

    end ; println(" ")

    println("> binary thermal hopfield:")

    for _ in 1:epochs

        temperature = temperature_initial

        states = binary_update_thermal(hopfield, input, temperature)
        grads = hebbian_grads(hopfield)
        hebbian_learn(hopfield, grads, lr)
        temperature *= temperature_decay

        @show states

    end ; println(" ")

    println("> binary thermal boltzmann:")

    for _ in 1:epochs

        temperature = temperature_initial

        states = binary_update_thermal_hiddens(boltzmann, input, temperature)
        grads = hebbian_grads(boltzmann)
        hebbian_learn(boltzmann, grads, lr)
        temperature *= temperature_decay

        @show states

    end ; println(" ")

    println("> continuous thermal boltzmann:")

    for _ in 1:epochs

        temperature = temperature_initial

        states = continuous_update_thermal_hiddens(boltzmann, input, temperature)
        grads = hebbian_grads(boltzmann)
        hebbian_learn(boltzmann, grads, lr)
        temperature *= temperature_decay

        @show states

    end ; println(" ")

end

## IMPORTS


using Random: shuffle, rand
using Knet: sigm


## PARAMS


max_update_steps          = 200
boltzmann_full_restricted = false
debug                     = true


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
                edge2 = Edge_Sym(nodes[j], nodes[i])
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

    Boltzmann(in_size, hidden_size) = begin

        visibles = [Node() for _ in 1:in_size]
        hiddens = [Node() for _ in 1:hidden_size]
        edges = []

        for i in 1:in_size
            for j in 1:hidden_size
                edge1 = Edge_Sym(visibles[i], hiddens[j])
                edge2 = Edge_Sym(hiddens[j], visibles[i])
                edge1.edge_sym = edge2
                edge2.edge_sym = edge1
                push!(edges, edge1)
                push!(edges, edge2)
                push!(visibles[i].edges, edge1)
                push!(hiddens[j].edges, edge2)
            end
        end

        if !boltzmann_full_restricted

            for i in 1:hidden_size
                for j in i+1:hidden_size
                    edge1 = Edge_Sym(hiddens[i], hiddens[j])
                    edge2 = Edge_Sym(hiddens[j], hiddens[i])
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
            return current_states
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
            randn() <= prob ? node.state = 1 : node.state = -1
        end

        current_states = [node.state for node in hopfield.nodes]

        if current_states == last_states
            return current_states
        else
            last_states = current_states
            ctr +=1
        end

    end

last_states
end


    # Binary Boltzmann


binary_update_thermal_hiddens(boltzmann::Boltzmann, input, temperature; hm_average=10, update_steps=max_update_steps) =
begin

    for (node, inp) in zip(boltzmann.visibles, input)
        node.state = inp
    end

    for node in boltzmann.hiddens
        node.state = round(rand())
    end

    last_states, ctr = nothing, 0

    while ctr <= update_steps

        for node in shuffle(boltzmann.hiddens)
            prob = sigm(sum([edge.weight * edge.node_to.state for edge in node.edges])/temperature)
            randn() <= prob ? node.state = 1 : node.state = -1
        end

        current_states = [node.state for node in boltzmann.hiddens]

        if current_states == last_states
            return last_states
        else
            last_states = current_states
            ctr +=1
        end

    end

last_states
end

binary_update_thermal_visibles(boltzmann::Boltzmann, hiddens, temperature; hm_average=10, update_steps=max_update_steps) =
begin

    for (node, hidden) in zip(boltzmann.hiddens, hiddens)
        node.state = hidden
    end

    for node in boltzmann.input
        node.state = round(rand())
    end

    last_states, ctr = nothing, 0

    while ctr <= update_steps

        for node in shuffle(boltzmann.visibles)
            prob = sigm(sum([edge.weight * edge.node_to.state for edge in node.edges])/temperature)
            randn() <= prob ? node.state = 1 : node.state = -1
        end

        current_states = [node.state for node in boltzmann.visibles]

        if current_states == last_states
            return last_states
        else
            last_states = current_states
            ctr +=1
        end

    end

last_states
end


    # Continuous Boltzmann


continuous_update_thermal_hiddens(boltzmann::Boltzmann, input, temperature; hm_average=10, update_steps=max_update_steps) =
begin

    for (node, inp) in zip(boltzmann.visibles, input)
        node.state = inp
    end

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
            return last_states
        else
            last_states = current_states
            ctr +=1
        end

    end

last_states
end

continuous_update_thermal_visibles(boltzmann::Boltzmann, hiddens, temperature; hm_average=10, update_steps=max_update_steps) =
begin

    for (node, hidden) in zip(boltzmann.visibles, hiddens)
        node.state = hidden
    end

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
            return last_states
        else
            last_states = current_states
            ctr +=1
        end

    end

last_states
end


    # Hebbian Learners


hebbian_grads(hopfield::Hopfield) =
begin

[edge.node_from.state * edge.node_to.state for edge in hopfield.edges]
end

hebbian_grads(boltzmann::Boltzmann) =
begin

[edge.node_from.state * edge.node_to.state for edge in boltzmann.visibles], [edge.node_from.state * edge.node_to.state for edge in boltzmann.hiddens]
end


hebbian_learn(hopfield::Hopfield, lr; grads=nothing) =

    if grads == nothing

        for edge in hopfield.edges
            edge.weight += lr * edge.node_from.state * edge.node_to.state
        end

    else

        for (edge, grad) in zip(hopfield.edges, grads)
            edge.weight += lr * grad
        end

    end

hebbian_learn(boltzmann::Boltzmann, lr; grads=nothing) =

    if grads == nothing

        for edge in boltzmann.edges
            edge.weight += lr * edge.node_from.state * edge.node_to.state
        end

    else

        for (edge, grad) in zip(boltzmann.edges, grads)
            edge.weight += lr * grad
        end

    end





## TESTS


input = [-1,1,-1,1,1,-1,-1]

# hopfield = Hopfield(length(input))
# states = binary_update_thermal(hopfield, input, 1.5) # binary_update(hopfield, input)
# @show states
# hebbian_learn(hopfield, .01)
# states = binary_update_thermal(hopfield, input, 1.25) # binary_update(hopfield, input)
# @show states
# hebbian_learn(hopfield, .01)
# states = binary_update_thermal(hopfield, input, 1.0) # binary_update(hopfield, input)
# @show states


main() = begin

    boltzmann = Boltzmann(length(input), 12)

    n = 10
    temp = 1.5
    temp_decay = .8

    for _ in 1:10
        states = binary_update_thermal_hiddens(boltzmann, input, temp, hm_average=10)
        @show states
        hebbian_learn(boltzmann, .01)
        temp *= temp_decay
    end

end ; main()

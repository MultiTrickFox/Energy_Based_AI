## IMPORTS


using Random: shuffle, rand
using Knet: sigm


## PARAMS


max_update_steps = 200

boltzmann_semi_restricted = true

debug = true


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

        if boltzmann_semi_restricted

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


## TRAINING HELPERS


binary_update(hopfield::Hopfield, input) =
begin

    for (node, inp) in zip(hopfield.nodes, input)
        node.state = inp
    end

    last_states, ctr = nothing, 0

    while ctr <= max_update_steps

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

binary_update_thermal(hopfield::Hopfield, input, temperature) =
begin

    for (node, inp) in zip(hopfield.nodes, input)
        node.state = inp
    end

    last_states, ctr = nothing, 0

    while ctr <= max_update_steps

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

binary_update_thermal(boltzmann::Boltzmann, input, temperature; hm_average=10) =
begin

    for (node, inp) in zip(boltzmann.visibles, input)
        node.state = inp
    end

    for node in boltzmann.hiddens
        node.state = randn()
    end

    last_states, ctr = nothing, 0

    while ctr <= max_update_steps

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

hebbian_learn(hopfield::Hopfield, lr) =
begin

    for edge in hopfield.edges
        edge.weight += lr * edge.node_from.state * edge.node_to.state
    end

end

hebbian_learn(boltzmann::Boltzmann, lr) =
begin

    for edge in boltzmann.edges
        edge.weight += lr * edge.node_from.state * edge.node_to.state
    end

end

hebbian_grads(hopfield::Hopfield) =
begin

[edge.node_from.state * edge.node_to.state for edge in hopfield.edges]
end

hebbian_grads(boltzmann::Boltzmann) =
begin

[edge.node_from.state * edge.node_to.state for edge in boltzmann.visibles], [edge.node_from.state * edge.node_to.state for edge in boltzmann.hiddens]
end


## GENERAL HELPERS


get_edge(node_from, node_to) =
begin
    for edge in node_from.edges
        if edge.node_to == node_to
            return edge
        end
    end
end

get_edge(node_from, node_to, hopfield) =
begin
    for edge in hopfield.edges
        if edge.node_from == node_from && edge.node_to == node_to
            return edge
        end
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
        states = binary_update_thermal(boltzmann, input, temp, hm_average=10)
        @show states
        hebbian_learn(boltzmann, .01)
        temp *= temp_decay
    end

end ; main()

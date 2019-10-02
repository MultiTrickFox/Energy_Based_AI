## IMPORTS


using Random: shuffle, rand
using Knet: sigm


## PARAMS


max_update_steps = 100


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


## TRAINING HELPERS


binary_update(hopfield::Hopfield, input) =
begin

    for (node, inp) in zip(hopfield.nodes, input)
        node.state = inp
    end

    last_states = nothing
    ctr = 0

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

    last_states = nothing
    ctr = 0

    while ctr <= max_update_steps

        for node in shuffle(hopfield.nodes)
            prob = sigm(sum([edge.weight * edge.node_to.state for edge in node.edges])/temperature)
            randn() <= prob ? node.state = 1 : node.state = 0
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

hebbian_learn(hopfield::Hopfield, lr) =
begin

    for edge in hopfield.edges
        edge.weight += lr * edge.node_from.state * edge.node_to.state
    end

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




input = [-1,1,-1,1,1]
hopfield = Hopfield(5)
states = binary_update(hopfield, input)
@show states

import os
from causallearn.utils.cit import CIT
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.Endpoint import Endpoint
import networkx as nx
import numpy as np
import csv
import pandas as pd

def assign_weights(data_matrix: np.array, cg: CausalGraph):
    result_obj = CIT(data_matrix, method='kci')  # Change `method` as needed
    undirected = cg.find_undirected()
    directed = cg.find_fully_directed()
    bidirected = cg.find_bi_directed()
    print(f'undirected {len(undirected)} directed {len(directed)} bidirected {len(bidirected)}')
    # Validate edges in the causal graph using Conditional Independence Tests (CIT)
    weights = np.zeros(cg.G.graph.shape)
    # Iterate through edges in the converted graph
    for (from_node, to_node) in directed:
        # Test conditional independence using CIT

        X=from_node
        Y=to_node
        # Extract the p-value from the result
        p_value = result_obj(X, Y, {})  # Adjust key if needed based on the CIT response structure
        # print(f"Edge: {from_node} -> {to_node}, p-value: {p_value}")

        # Retain edge if the conditional independence test rejects independence (low p-value)
        # if p_value < 0.05:  # Use a threshold (e.g., 0.05) to determine significance
        # set(cg.G.get_edge(from_node, to_node), 'weight', p_value)
        # print(f"Edge: {from_node} -> {to_node}, p-value: {p_value}")
        weights[int(from_node), int(to_node)] = p_value
    setattr(cg, 'weight', weights)
    return cg

def save_weight_csv(cg: CausalGraph, weight_thresh=0.05, file_name='weights.csv', data_dir=None):
    edges = None
    nodes = cg.G.nodes
    weights = []
    if edges is None:
        edges = cg.G.get_graph_edges()
        for edge in edges:
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            node1_id = nodes.index(node1)
            node2_id = nodes.index(node2)
            if(cg.weight[int(node1_id), int(node2_id)]>weight_thresh
               or cg.weight[int(node1_id), int(node2_id)]==0):
                continue
            else:
                weights.append([cg.weight[int(node1_id), int(node2_id)], node1.get_name(), node2.get_name()])
    # sort weights according to the first element
    weights.sort(key=lambda x: x[0])
    df = pd.DataFrame(weights, columns=["Value", "Name1", "Name2"])
    df["Value"] = df["Value"].astype(float)
    if data_dir is not None:
        if data_dir.startswith('~/'):
            data_dir = os.path.expanduser(data_dir)
        elif data_dir.startswith('./'):
            data_dir = os.path.abspath(data_dir)
        assert(os.path.isdir(data_dir)), f"Directory {data_dir} does not exist"
        file_name = os.path.join(data_dir, file_name)
    
    print(f"Saving weights to {file_name}")
    df.to_csv(file_name, index=False)
        
def create_nx_graph(graph, node_idx, edge_idx):
    subgraph = nx.DiGraph()
    nodes = [graph.nodes[i] for i in node_idx]
    subgraph.add_nodes_from(nodes)
    for (i, j) in edge_idx:
        subgraph.add_edge(graph.nodes[i], graph.nodes[j], color='r')
    return subgraph

def extract_subgraph_direct(graph, target_node_idx, level=1):
    print(f'number of nodes {(graph.get_num_nodes())}')
    print(f'number of edges {(graph.get_num_edges())}')
    candidate_idx = [i for i in range(graph.get_num_nodes()) if i != target_node_idx]
    # Initialize subgraph
    subgraph_nodes = {}
    subgraph_nodes[target_node_idx] = graph.nodes[target_node_idx]
    subgraph_edges = []
    for i in candidate_idx:
        if (graph.graph[i, target_node_idx] == -1 and graph.graph[target_node_idx, i] == 1) \
                    or (graph.graph[i, target_node_idx] == Endpoint.TAIL_AND_ARROW.value
                        and graph.graph[target_node_idx, i] == Endpoint.ARROW_AND_ARROW.value):
            subgraph_nodes[i] = graph.nodes[i].name
            subgraph_edges.append((i, target_node_idx))
        elif (graph.graph[i, target_node_idx] == 1 and graph.graph[target_node_idx, i] == -1) \
                    or (graph.graph[target_node_idx, i] == Endpoint.TAIL_AND_ARROW.value
                        and graph.graph[i, target_node_idx ] == Endpoint.ARROW_AND_ARROW.value):
            subgraph_nodes[i] = graph.nodes[i].name
            subgraph_edges.append((target_node_idx,i))

    print(f'number of nodes in subgraph {(len(subgraph_nodes))}')
    print(f'number of edges in subgraph {(len(subgraph_edges))}')
    return subgraph_nodes, subgraph_edges

def extract_subgraph(graph, target_node_idx):
    print(f'number of nodes {(graph.get_num_nodes())}')
    print(f'number of edges {(graph.get_num_edges())}')
    candidate_idx = [i for i in range(graph.get_num_nodes()) if i != target_node_idx]
    # Initialize subgraph
    subgraph_nodes = {}
    subgraph_edges = []
    # Use a stack for iterative parent traversal
    stack_idx = [target_node_idx] # id: label
    idx_explored = [target_node_idx] #[id1,id2]

    while stack_idx:
        current_idx = stack_idx.pop()
        # Avoid processing a node more than once
        if graph.nodes[current_idx] not in subgraph_nodes:
            subgraph_nodes[current_idx] = graph.nodes[current_idx].name
        for i in candidate_idx:
            if (graph.graph[i, current_idx] == -1 and graph.graph[current_idx, i] == 1) \
                    or (graph.graph[i, current_idx] == Endpoint.TAIL_AND_ARROW.value
                        and graph.graph[current_idx, i] == Endpoint.ARROW_AND_ARROW.value):
                subgraph_nodes[i] = graph.nodes[i].name
                subgraph_edges.append((i, current_idx))
                if i not in idx_explored:
                    stack_idx.append(i)
                    idx_explored.append(i)

    print(f'number of nodes in subgraph {(len(subgraph_nodes))}')
    print(f'number of edges in subgraph {(len(subgraph_edges))}')
    return subgraph_nodes, subgraph_edges

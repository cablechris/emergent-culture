#!/usr/bin/env python3

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from environment import run_simulation
from collections import defaultdict, Counter

def build_trait_lineage_graph(log):
    """
    Build a directed graph representing trait lineage.
    
    Args:
        log: Dictionary mapping epochs to lists of agent state dictionaries
        
    Returns:
        NetworkX DiGraph with trait lineage relationships
    """
    G = nx.DiGraph()
    
    # Process all epochs in the log
    for epoch, epoch_data in log.items():
        for agent in epoch_data:
            tid = agent["trait_id"]
            pid = agent["parent_trait_id"]
            
            # Add current trait node if it's new
            if tid not in G:
                # Extract the first 8 chars of UUID for display
                short_id = tid[:8]
                G.add_node(tid, label=short_id, trait=agent["trait"], epoch_observed=epoch)
            
            # Add parent-child relationship if parent exists
            if pid and pid not in G:
                # Add parent node (may have been from earlier epoch)
                short_pid = pid[:8]
                G.add_node(pid, label=short_pid)
            
            # Add edge from parent to child
            if pid and pid != tid:  # Avoid self-loops
                G.add_edge(pid, tid)
    
    return G

def visualize_trait_lineage(G, show_labels=False, min_node_count=1):
    """
    Visualize the trait lineage graph.
    
    Args:
        G: NetworkX DiGraph of trait lineage
        show_labels: Whether to show node labels
        min_node_count: Minimum degree to show labels (if show_labels is True)
    """
    plt.figure(figsize=(16, 12))
    
    # Use hierarchical layout
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Determine node sizes based on their out-degree (number of offspring)
    node_sizes = []
    labels = {}
    
    for node in G.nodes():
        # Node size based on number of descendants
        descendants = len(nx.descendants(G, node))
        node_sizes.append(50 + descendants * 10)
        
        # Only show labels for nodes with enough descendants
        if show_labels and descendants >= min_node_count:
            labels[node] = G.nodes[node].get('label', str(node)[:8])
    
    # Create a colormap based on depth in the tree
    root_nodes = [n for n, d in G.in_degree() if d == 0]
    node_depth = {}
    
    for root in root_nodes:
        for node in nx.dfs_preorder_nodes(G, root):
            # Assign depth based on shortest path from any root
            if node not in node_depth:
                node_depth[node] = nx.shortest_path_length(G, root, node)
            else:
                node_depth[node] = min(node_depth[node], 
                                      nx.shortest_path_length(G, root, node))
    
    # Convert depths to colors
    max_depth = max(node_depth.values()) if node_depth else 0
    node_colors = []
    
    for node in G.nodes():
        if node in node_depth:
            depth = node_depth[node]
            # Use a color gradient from red (new) to blue (old)
            color = plt.cm.cool(depth / max_depth if max_depth > 0 else 0)
            node_colors.append(color)
        else:
            node_colors.append('gray')
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, arrows=True, width=0.8, 
                          arrowsize=10, connectionstyle='arc3,rad=0.1')
    
    if show_labels:
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    plt.title("Trait Lineage Network", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def analyze_trait_lineage(G):
    """
    Analyze the trait lineage graph and print statistics.
    
    Args:
        G: NetworkX DiGraph of trait lineage
    """
    # Count the number of unique traits
    num_traits = G.number_of_nodes()
    
    # Find root traits (original traits with no parents)
    root_traits = [n for n, d in G.in_degree() if d == 0]
    
    # Find leaf traits (traits with no offspring)
    leaf_traits = [n for n, d in G.out_degree() if d == 0]
    
    # Find the most successful traits (those with the most descendants)
    trait_success = {node: len(nx.descendants(G, node)) for node in G.nodes()}
    most_successful = sorted(trait_success.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate the average number of offspring per trait
    avg_offspring = sum(G.out_degree(n) for n in G.nodes()) / max(1, num_traits)
    
    # Calculate the maximum tree depth
    max_depth = 0
    for root in root_traits:
        for node in nx.descendants(G, root):
            path_length = len(nx.shortest_path(G, root, node)) - 1
            max_depth = max(max_depth, path_length)
    
    # Print the statistics
    print("\nTrait Lineage Analysis:")
    print(f"Total unique traits: {num_traits}")
    print(f"Original traits (roots): {len(root_traits)}")
    print(f"Terminal traits (leaves): {len(leaf_traits)}")
    print(f"Maximum lineage depth: {max_depth}")
    print(f"Average offspring per trait: {avg_offspring:.2f}")
    print("\nMost successful traits (by number of descendants):")
    for i, (trait, descendants) in enumerate(most_successful[:10]):
        short_id = trait[:8]
        print(f"  {i+1}. Trait {short_id}: {descendants} descendants")

def main():
    """Run a simulation and visualize trait lineage"""
    
    # Check if we have existing trait lineage data
    if os.path.exists('trait_lineage_data.pkl'):
        print("Loading existing trait lineage data...")
        with open('trait_lineage_data.pkl', 'rb') as f:
            lineage_log = pickle.load(f)
    else:
        print("Running new simulation to gather trait lineage data...")
        # Run a simulation with 100 epochs
        _, _, _, _, _, _, lineage_log = run_simulation(
            num_agents=50,
            num_epochs=100,
            use_behaviors=True,
            selective_imitation=True
        )
        
        # Save the lineage data
        with open('trait_lineage_data.pkl', 'wb') as f:
            pickle.dump(lineage_log, f)
    
    # Build the trait lineage graph
    print("Building trait lineage graph...")
    lineage_graph = build_trait_lineage_graph(lineage_log)
    
    # Analyze the trait lineage
    analyze_trait_lineage(lineage_graph)
    
    # Visualize the trait lineage
    print("\nVisualizing trait lineage network...")
    visualize_trait_lineage(lineage_graph, show_labels=True, min_node_count=5)

if __name__ == "__main__":
    main() 
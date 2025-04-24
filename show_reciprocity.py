#!/usr/bin/env python3

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from environment import run_simulation
from reciprocity_utils import get_reciprocity_score, calculate_network_reciprocity, identify_reciprocal_pairs

def plot_reciprocity_network(agents, threshold=1):
    """
    Visualize the network of reciprocal relationships between agents.
    
    Args:
        agents: List of Agent objects
        threshold: Minimum reciprocity score to include an edge
    """
    G = nx.Graph()
    
    # Add all agents as nodes
    for agent in agents:
        G.add_node(agent.id)
    
    # Add edges for reciprocal relationships
    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            reciprocity = get_reciprocity_score(a1, a2)
            if reciprocity >= threshold:
                G.add_edge(a1.id, a2.id, weight=reciprocity)
    
    # Calculate node sizes based on total reciprocity
    node_sizes = []
    for agent in agents:
        total_reciprocity = sum(get_reciprocity_score(agent, other) 
                             for other in agents if other.id != agent.id)
        node_sizes.append(100 + total_reciprocity * 20)
    
    # Calculate edge widths based on reciprocity weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes with sizes based on reciprocity
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='skyblue', alpha=0.8)
    
    # Draw edges with width based on reciprocity weight
    nx.draw_networkx_edges(G, pos, width=[w/2 for w in edge_weights], 
                          edge_color='gray', alpha=0.6)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title(f"Agent Reciprocity Network (threshold={threshold})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_reciprocity_distribution(agents):
    """
    Plot the distribution of reciprocity scores across all agent pairs.
    
    Args:
        agents: List of Agent objects
    """
    all_scores = []
    
    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            score = get_reciprocity_score(a1, a2)
            all_scores.append(score)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=max(10, max(all_scores)+1), alpha=0.7)
    plt.title("Distribution of Reciprocity Scores Between Agent Pairs")
    plt.xlabel("Reciprocity Score")
    plt.ylabel("Number of Agent Pairs")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_top_reciprocal_pairs(agents, top_n=10):
    """
    Visualize the top reciprocal agent pairs.
    
    Args:
        agents: List of Agent objects
        top_n: Number of top pairs to display
    """
    reciprocal_pairs = identify_reciprocal_pairs(agents)
    
    # Take top N pairs
    top_pairs = reciprocal_pairs[:min(top_n, len(reciprocal_pairs))]
    
    if not top_pairs:
        print("No reciprocal pairs found.")
        return
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    pair_labels = [f"Agents {a}-{b}" for a, b, _ in top_pairs]
    scores = [score for _, _, score in top_pairs]
    
    y_pos = np.arange(len(pair_labels))
    ax.barh(y_pos, scores, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pair_labels)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Reciprocity Score')
    ax.set_title(f'Top {len(top_pairs)} Agent Pairs by Reciprocity')
    
    plt.tight_layout()
    plt.show()

# Check if we have existing data, otherwise run a new simulation
if os.path.exists('dual_signal_data.pkl'):
    print("Loading saved dual-signal simulation data...")
    with open('dual_signal_data.pkl', 'rb') as f:
        data = pickle.load(f)
        if len(data) >= 6:  # Should have all the data we need
            _, final_agents, _, _, _, _ = data
        else:
            print("Old data format detected, running new simulation...")
            _, final_agents, _, _, _, _ = run_simulation(
                use_behaviors=True,
                selective_imitation=True,
                num_epochs=100
            )
else:
    print("Running new dual-signal simulation...")
    # Run a simulation with behaviors enabled
    _, final_agents, _, _, _, _ = run_simulation(
        use_behaviors=True,
        selective_imitation=True,
        num_epochs=100
    )
    # We don't save here as it will be saved by the other visualization scripts

# Calculate overall network reciprocity
network_reciprocity = calculate_network_reciprocity(final_agents)
print(f"\nOverall Network Reciprocity: {network_reciprocity:.4f}")

print("\nVisualization 1: Reciprocity Network")
plot_reciprocity_network(final_agents, threshold=1)

print("\nVisualization 2: Reciprocity Distribution")
plot_reciprocity_distribution(final_agents)

print("\nVisualization 3: Top Reciprocal Pairs")
plot_top_reciprocal_pairs(final_agents) 
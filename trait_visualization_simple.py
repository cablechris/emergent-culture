#!/usr/bin/env python3

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter

def load_trait_lineage_data():
    """Load trait lineage data from file"""
    if os.path.exists('trait_lineage_data.pkl'):
        print("Loading trait lineage data...")
        with open('trait_lineage_data.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        print("No trait lineage data found. Please run trait_lineage.py first.")
        return None

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

def visualize_small_lineages(G, max_lineages=5, max_nodes_per_lineage=20):
    """
    Visualize several small trait lineages for clarity.
    
    Args:
        G: NetworkX DiGraph of trait lineage
        max_lineages: Maximum number of lineages to show
        max_nodes_per_lineage: Maximum nodes per lineage
    """
    # Find root nodes
    root_nodes = [n for n, d in G.in_degree() if d == 0]
    
    # Get the most "successful" roots (those with most descendants)
    root_success = [(root, len(nx.descendants(G, root))) for root in root_nodes]
    top_roots = sorted(root_success, key=lambda x: x[1], reverse=True)[:max_lineages]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(top_roots), figsize=(16, 6), 
                             gridspec_kw={'width_ratios': [max(1, d) for _, d in top_roots]})
    if len(top_roots) == 1:
        axes = [axes]  # Make it a list for consistency
    
    # Plot each lineage in its own subplot
    for i, (root, desc_count) in enumerate(top_roots):
        if desc_count == 0:
            continue  # Skip roots with no descendants
            
        # Get descendants
        descendants = list(nx.descendants(G, root))
        
        # Take a sample if too many
        if len(descendants) > max_nodes_per_lineage:
            descendants = np.random.choice(descendants, max_nodes_per_lineage, replace=False).tolist()
        
        # Create subgraph with this root and its (sampled) descendants
        nodes = [root] + descendants
        subG = G.subgraph(nodes)
        
        # Create a separate hierarchical layout for this subgraph
        pos = nx.spring_layout(subG, seed=i+42)
        
        # Use a different color for each level in the hierarchy
        node_colors = []
        node_sizes = []
        
        # Try to compute shortest path lengths from root
        try:
            node_depths = nx.shortest_path_length(subG, root)
            max_depth = max(node_depths.values()) if node_depths else 0
            
            for node in subG.nodes():
                depth = node_depths.get(node, 0)
                # Use a color gradient from green (root) to red (leaves)
                color = plt.cm.viridis(depth / max(1, max_depth))
                node_colors.append(color)
                
                # Size nodes by their number of descendants
                descendants = len(list(nx.descendants(subG, node)))
                node_sizes.append(100 + descendants * 10)
        except:
            # Fallback if path computation fails
            node_colors = ['skyblue'] * len(subG.nodes())
            node_sizes = [100] * len(subG.nodes())
            
        # Draw the subgraph
        nx.draw_networkx_nodes(subG, pos, node_color=node_colors, 
                              node_size=node_sizes, ax=axes[i], alpha=0.8)
        nx.draw_networkx_edges(subG, pos, edge_color='gray', 
                              alpha=0.6, width=1.0, ax=axes[i], 
                              arrows=True, arrowsize=10)
        
        # Label the root and a few key nodes
        labels = {root: G.nodes[root].get('label', root[:8])}
        
        # Add a few other important node labels
        important_nodes = sorted([(n, len(list(nx.descendants(subG, n)))) 
                                 for n in subG.nodes() if n != root],
                                key=lambda x: x[1], reverse=True)[:3]
        for node, _ in important_nodes:
            labels[node] = G.nodes[node].get('label', node[:8])
            
        nx.draw_networkx_labels(subG, pos, labels=labels, 
                               font_size=8, ax=axes[i])
        
        axes[i].set_title(f"Lineage {i+1}: Root {G.nodes[root].get('label', root[:8])}\n({desc_count} descendants)")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_trait_evolution_metrics(lineage_log):
    """
    Plot metrics about trait evolution over time.
    
    Args:
        lineage_log: Dictionary mapping epochs to lists of agent state dictionaries
    """
    # Extract epochs
    epochs = sorted(lineage_log.keys())
    
    # Metrics to track
    unique_traits_count = []
    trait_diversity = []
    trait_turnover = []
    
    # Previous epoch traits for turnover calculation
    prev_traits = set()
    
    for epoch in epochs:
        # Get all traits in this epoch
        traits = [agent['trait'] for agent in lineage_log[epoch]]
        unique_traits = set(traits)
        
        # Count unique traits
        unique_traits_count.append(len(unique_traits))
        
        # Calculate Shannon entropy for trait diversity
        trait_counter = Counter(traits)
        total = len(traits)
        entropy = 0
        for count in trait_counter.values():
            p = count / total
            entropy -= p * np.log2(p)
        trait_diversity.append(entropy)
        
        # Calculate trait turnover (Jaccard distance)
        if prev_traits:
            intersection = len(unique_traits.intersection(prev_traits))
            union = len(unique_traits.union(prev_traits))
            turnover = 1 - (intersection / union) if union > 0 else 0
            trait_turnover.append(turnover)
        elif len(epochs) > 1:  # Add a placeholder for the first epoch
            trait_turnover.append(0)
            
        # Update previous traits
        prev_traits = unique_traits
    
    # Create the plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Unique trait count
    axes[0].plot(epochs, unique_traits_count, 'b-o')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Number of Unique Traits Over Time')
    axes[0].grid(alpha=0.3)
    
    # Trait diversity (Shannon entropy)
    axes[1].plot(epochs, trait_diversity, 'g-s')
    axes[1].set_ylabel('Shannon Entropy')
    axes[1].set_title('Trait Diversity Over Time')
    axes[1].grid(alpha=0.3)
    
    # Trait turnover
    axes[2].plot(epochs, trait_turnover, 'r-^')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Jaccard Distance')
    axes[2].set_title('Trait Turnover Between Epochs')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_trait_age_distribution(G):
    """
    Visualize the distribution of trait ages (number of descendants)
    
    Args:
        G: NetworkX DiGraph of trait lineage
    """
    # Calculate number of descendants for each trait
    trait_descendants = []
    for node in G.nodes():
        num_descendants = len(list(nx.descendants(G, node)))
        trait_descendants.append(num_descendants)
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(trait_descendants, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Number of Descendant Traits')
    plt.ylabel('Count')
    plt.title('Distribution of Trait Evolutionary Success')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"Total traits: {len(trait_descendants)}")
    print(f"Max descendants: {max(trait_descendants)}")
    print(f"Mean descendants: {np.mean(trait_descendants):.2f}")
    print(f"Median descendants: {np.median(trait_descendants):.2f}")
    
    # Count traits with no descendants (extinct)
    extinct = sum(1 for d in trait_descendants if d == 0)
    print(f"Extinct traits (no descendants): {extinct} ({extinct/len(trait_descendants)*100:.1f}%)")

def main():
    """Run simple trait lineage visualizations"""
    
    # Load data
    lineage_log = load_trait_lineage_data()
    if not lineage_log:
        return
        
    # Build trait lineage graph
    print("Building trait lineage graph...")
    G = build_trait_lineage_graph(lineage_log)
    
    # Create visualizations
    print("\n1. Small Trait Lineages")
    visualize_small_lineages(G)
    
    print("\n2. Trait Evolution Metrics")
    plot_trait_evolution_metrics(lineage_log)
    
    print("\n3. Trait Age Distribution")
    visualize_trait_age_distribution(G)

if __name__ == "__main__":
    main() 
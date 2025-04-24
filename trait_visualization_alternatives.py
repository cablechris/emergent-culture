#!/usr/bin/env python3

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
import matplotlib.cm as cm
from matplotlib.colors import Normalize
# Import seaborn if available, otherwise use matplotlib
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not installed. Some visualizations will use matplotlib instead.")
    print("To install: pip install seaborn")

from trait_lineage import build_trait_lineage_graph

def visualize_radial_tree(G):
    """
    Visualize the trait lineage as a radial/circular tree layout.
    
    Args:
        G: NetworkX DiGraph of trait lineage
    """
    plt.figure(figsize=(16, 16))
    
    # Find the root nodes (nodes with no parents)
    root_nodes = [n for n, d in G.in_degree() if d == 0]
    
    # Create a layout with the largest connected component
    if not root_nodes:
        print("No root nodes found, using random node as root")
        root = list(G.nodes())[0] if G.nodes() else None
    else:
        # Use the root with the most descendants
        root = max(root_nodes, key=lambda n: len(nx.descendants(G, n)) if nx.has_path(G, n, list(G.nodes())[-1]) else 0)
    
    if not root:
        print("No valid root found for visualization")
        return
    
    # Get all nodes reachable from the root
    reachable = nx.descendants(G, root)
    reachable.add(root)
    
    # Create a subgraph with the root and its descendants
    subG = G.subgraph(reachable)
    
    # Create layout - try graphviz first, fall back to networkx layouts
    try:
        pos = nx.drawing.nx_agraph.graphviz_layout(subG, prog="twopi", root=root)
    except (ImportError, Exception) as e:
        print(f"Graphviz layout failed ({str(e)}), falling back to spring layout")
        print("To use radial layout, install: pip install pygraphviz")
        pos = nx.spring_layout(subG, center=[0, 0], scale=1000, seed=42)
    
    # Color nodes by their depth from the root
    node_depths = nx.shortest_path_length(subG, root)
    max_depth = max(node_depths.values()) if node_depths else 0
    node_colors = [plt.cm.viridis(node_depths.get(n, 0) / max_depth) for n in subG.nodes()]
    
    # Scale node sizes by number of descendants
    node_sizes = []
    for node in subG.nodes():
        descendants = len(list(nx.descendants(subG, node)))
        node_sizes.append(50 + descendants * 5)
    
    # Draw the graph
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(subG, pos, edge_color='gray', alpha=0.4, width=0.6, 
                           arrowsize=5, arrows=True)
    
    # Optionally add labels for important nodes (those with many descendants)
    labels = {n: n[:8] for n in subG.nodes() if len(list(nx.descendants(subG, n))) > 5}
    nx.draw_networkx_labels(subG, pos, labels=labels, font_size=8)
    
    plt.title(f"Trait Lineage - {root[:8]} Lineage", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_trait_popularity_heatmap(lineage_log):
    """
    Create a heatmap showing the popularity of traits over time.
    
    Args:
        lineage_log: Dictionary mapping epochs to lists of agent state dictionaries
    """
    # Track trait counts by epoch
    trait_counts_by_epoch = {}
    all_traits = set()
    
    for epoch, agents in lineage_log.items():
        # Count traits in this epoch
        trait_counter = Counter(agent['trait'] for agent in agents)
        trait_counts_by_epoch[epoch] = trait_counter
        all_traits.update(trait_counter.keys())
    
    # Sort epochs and get top traits
    epochs = sorted(trait_counts_by_epoch.keys())
    
    # Combine counts across all epochs to find the most common traits
    all_counts = Counter()
    for counter in trait_counts_by_epoch.values():
        all_counts.update(counter)
    
    # Get the top 20 most common traits
    top_traits = [trait for trait, _ in all_counts.most_common(20)]
    
    # Create the data matrix for the heatmap
    data = np.zeros((len(top_traits), len(epochs)))
    for i, trait in enumerate(top_traits):
        for j, epoch in enumerate(epochs):
            data[i, j] = trait_counts_by_epoch[epoch].get(trait, 0)
    
    # Plot the heatmap
    plt.figure(figsize=(14, 8))
    
    if HAS_SEABORN:
        # Use seaborn for a nicer heatmap
        sns.heatmap(data, cmap="YlGnBu", 
                    xticklabels=[str(e) for e in epochs], 
                    yticklabels=[f"{t[:8]}..." for t in top_traits],
                    cbar_kws={'label': 'Count'})
    else:
        # Fallback to matplotlib
        plt.imshow(data, aspect='auto', cmap='YlGnBu')
        plt.colorbar(label='Count')
        plt.xticks(range(len(epochs)), [str(e) for e in epochs])
        plt.yticks(range(len(top_traits)), [f"{t[:8]}..." for t in top_traits])
    
    plt.title("Top Trait Popularity Over Time", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Trait", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_trait_lineage_over_time(lineage_log, G):
    """
    Plot trait lineage metrics over time.
    
    Args:
        lineage_log: Dictionary mapping epochs to lists of agent state dictionaries
        G: NetworkX DiGraph of trait lineage
    """
    # Calculate metrics by epoch
    epochs = sorted(lineage_log.keys())
    unique_traits = []
    trait_diversity = []
    new_traits = []
    
    # Track which traits are observed in each epoch
    traits_by_epoch = {}
    all_traits_so_far = set()
    
    for epoch in epochs:
        agent_data = lineage_log[epoch]
        epoch_traits = set(agent['trait_id'] for agent in agent_data)
        traits_by_epoch[epoch] = epoch_traits
        
        # Count unique traits in this epoch
        unique_traits.append(len(epoch_traits))
        
        # Calculate trait diversity (Shannon entropy)
        trait_counter = Counter(agent['trait'] for agent in agent_data)
        total = sum(trait_counter.values())
        
        if total > 0:
            entropy = -sum((count/total) * np.log2(count/total) for count in trait_counter.values())
            trait_diversity.append(entropy)
        else:
            trait_diversity.append(0)
        
        # Count new traits in this epoch
        new_in_epoch = epoch_traits - all_traits_so_far
        new_traits.append(len(new_in_epoch))
        all_traits_so_far.update(epoch_traits)
    
    # Create a multi-panel plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Unique traits per epoch
    axes[0].plot(epochs, unique_traits, 'b-', marker='o', label="Unique Traits")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Unique Traits per Epoch")
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Trait diversity (Shannon entropy)
    axes[1].plot(epochs, trait_diversity, 'g-', marker='s', label="Trait Diversity")
    axes[1].set_ylabel("Shannon Entropy")
    axes[1].set_title("Trait Diversity Over Time")
    axes[1].grid(alpha=0.3)
    
    # Plot 3: New traits per epoch
    axes[2].bar(epochs, new_traits, color='orange', label="New Traits")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Count")
    axes[2].set_title("New Traits Introduced per Epoch")
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_trait_lineage_clusters(G):
    """
    Visualize clusters in the trait lineage.
    
    Args:
        G: NetworkX DiGraph of trait lineage
    """
    # Create an undirected copy of the graph for community detection
    G_undirected = G.to_undirected()
    
    # Detect communities
    try:
        import community as community_louvain
    except ImportError:
        try:
            import community.community_louvain as community_louvain
        except ImportError:
            print("Community detection requires the python-louvain package.")
            print("Install with: pip install python-louvain")
            return
    
    # Perform community detection
    partition = community_louvain.best_partition(G_undirected)
    
    # Get the set of communities
    communities = set(partition.values())
    
    # Calculate sizes of communities
    community_sizes = Counter(partition.values())
    
    # Get the largest communities
    largest_communities = [comm for comm, size in community_sizes.most_common() 
                          if size > 3]  # Only communities with more than 3 members
    
    print(f"Detected {len(communities)} communities")
    print(f"Largest community has {max(community_sizes.values())} traits")
    
    # Create a layout using spring layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Draw the graph with community colors
    plt.figure(figsize=(16, 12))
    
    # Define a colormap
    cmap = plt.cm.tab20
    norm = Normalize(vmin=0, vmax=max(communities))
    
    # Draw nodes with community colors
    for community in largest_communities:
        # Get nodes in the community
        community_nodes = [n for n, c in partition.items() if c == community]
        
        # Skip tiny communities
        if len(community_nodes) < 4:
            continue
            
        # Draw nodes for this community
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=community_nodes,
                              node_color=[cmap(norm(community))] * len(community_nodes),
                              node_size=80,
                              alpha=0.7,
                              label=f"Community {community}")
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray', 
                          width=0.3, 
                          alpha=0.3,
                          arrows=True,
                          arrowsize=5)
    
    plt.title("Trait Lineage Communities", fontsize=16)
    plt.axis('off')
    plt.legend(scatterpoints=1, loc='lower right')
    plt.tight_layout()
    plt.show()

def main():
    """Run alternative trait lineage visualizations"""
    
    # Check if we have existing trait lineage data
    if os.path.exists('trait_lineage_data.pkl'):
        print("Loading existing trait lineage data...")
        with open('trait_lineage_data.pkl', 'rb') as f:
            lineage_log = pickle.load(f)
            
        # Build the trait lineage graph
        print("Building trait lineage graph...")
        lineage_graph = build_trait_lineage_graph(lineage_log)
        
        # Create visualizations
        print("\n1. Radial Tree Visualization")
        visualize_radial_tree(lineage_graph)
        
        print("\n2. Trait Popularity Heatmap")
        plot_trait_popularity_heatmap(lineage_log)
        
        print("\n3. Trait Metrics Over Time")
        plot_trait_lineage_over_time(lineage_log, lineage_graph)
        
        print("\n4. Trait Lineage Communities")
        visualize_trait_lineage_clusters(lineage_graph)
    else:
        print("No trait lineage data found. Please run trait_lineage.py first.")

if __name__ == "__main__":
    main() 
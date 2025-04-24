#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import networkx as nx

def plot_cost_variant_adoption_over_time(log, variant_names=('trait_low_cost', 'trait_med_cost', 'trait_high_cost')):
    """
    Plots the number of agents using each trait variant (by name) over time.
    
    Args:
        log: Simulation log with epoch data
        variant_names: Tuple of variant names to track (with trait_ prefix)
    """
    variant_counts_by_epoch = {variant: [] for variant in variant_names}
    variant_counts_by_epoch['other'] = []  # Add 'other' category
    epochs = sorted(log.keys())

    for epoch in epochs:
        # Get all traits from this epoch
        traits = []
        for entry in log[epoch]:
            # Extract trait - could be at index 2 depending on log format
            if len(entry) >= 3:
                trait = entry[2]
                traits.append(trait)
        
        # Count variants
        counts = defaultdict(int)
        for trait in traits:
            variant_found = False
            for variant in variant_names:
                # Check if variant name is in the trait string
                if isinstance(trait, str) and variant in trait:
                    counts[variant] += 1
                    variant_found = True
                    break
            
            if not variant_found:
                counts['other'] += 1
        
        # Add to counts by epoch
        for variant in variant_names:
            variant_counts_by_epoch[variant].append(counts[variant])
        variant_counts_by_epoch['other'].append(counts['other'])

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot each variant
    all_variants = list(variant_names) + ['other']
    colors = ['green', 'blue', 'red', 'gray']
    
    for i, variant in enumerate(all_variants):
        color = colors[i % len(colors)]
        label = variant.replace('trait_', '') if variant != 'other' else variant
        plt.plot(epochs, variant_counts_by_epoch[variant], 
                 label=label, linewidth=2, color=color)

    plt.title("Adoption Over Time by Trait Cost Variant")
    plt.xlabel("Epoch")
    plt.ylabel("Number of Adopters")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotations for start and end values
    for i, variant in enumerate(all_variants):
        if variant_counts_by_epoch[variant]:
            start_val = variant_counts_by_epoch[variant][0]
            end_val = variant_counts_by_epoch[variant][-1]
            label = variant.replace('trait_', '')
            
            # Only annotate if values are significant
            if start_val > 1:
                plt.annotate(f"{start_val}", 
                             (epochs[0], start_val),
                             textcoords="offset points",
                             xytext=(0,10), 
                             ha='center')
            
            if end_val > 1:
                plt.annotate(f"{end_val}", 
                             (epochs[-1], end_val),
                             textcoords="offset points",
                             xytext=(0,10), 
                             ha='center')
    
    plt.tight_layout()
    plt.show()

def plot_trait_frequency_over_time(log, top_n=5):
    """
    Plots the frequency of the most common traits over time.
    
    Args:
        log: Simulation log with epoch data
        top_n: Number of top traits to track
    """
    epochs = sorted(log.keys())
    trait_counts = defaultdict(lambda: [0] * len(epochs))
    
    # Count trait frequencies for each epoch
    for i, epoch in enumerate(epochs):
        traits = [entry[2] for entry in log[epoch]]
        trait_counter = defaultdict(int)
        for trait in traits:
            trait_counter[trait] += 1
        
        # Update counts for all tracked traits
        for trait, count in trait_counter.items():
            trait_counts[trait][i] = count
    
    # Find the top N traits by their maximum frequency
    top_traits = sorted(trait_counts.items(), 
                        key=lambda x: max(x[1]), 
                        reverse=True)[:top_n]
    
    # Plot
    plt.figure(figsize=(12, 6))
    for trait, counts in top_traits:
        plt.plot(epochs, counts, label=f"{trait[:15]}..." if len(trait) > 15 else trait, linewidth=2)
    
    plt.title(f"Top {top_n} Trait Frequencies Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_recombination_impact(results, metric_name, title=None):
    """
    Plots a specific metric for different recombination rates.
    
    Args:
        results: Dictionary with recombination experiment results
        metric_name: Name of the metric to plot
        title: Optional custom title
    """
    rates = sorted(results.keys())
    metrics = []
    
    for rate in rates:
        data = results[rate]
        agents = data['agents']
        
        # Get the appropriate metric based on the name
        if metric_name == 'trait_diversity':
            # Count unique traits
            traits = [''.join(agent.trait) for agent in agents]
            metrics.append(len(set(traits)))
        elif metric_name == 'trait_complexity':
            # Average trait complexity (entropy)
            from scipy.stats import entropy
            traits = [''.join(agent.trait) for agent in agents]
            complexities = []
            for trait in set(traits):
                char_counts = {}
                for char in trait:
                    char_counts[char] = char_counts.get(char, 0) + 1
                probabilities = [count/len(trait) for count in char_counts.values()]
                complexities.append(entropy(probabilities))
            metrics.append(np.mean(complexities) if complexities else 0)
        elif metric_name == 'recombined_proportion':
            # Proportion of recombined traits
            recombined = sum(1 for agent in agents if agent.trait_meta.get('recombined', False))
            metrics.append(recombined / len(agents) * 100)
        else:
            # Try to extract the metric from the first agent's metadata
            metrics.append(0)  # Default if not found
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(rates)), metrics, color='skyblue')
    plt.xticks(range(len(rates)), [str(r) for r in rates])
    plt.xlabel('Recombination Rate')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(title or f"Impact of Recombination Rate on {metric_name.replace('_', ' ').title()}")
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_trait_network(agents, threshold=0.5, label_nodes=True):
    """
    Plots a network of traits where edges represent similarity.
    
    Args:
        agents: List of agents
        threshold: Similarity threshold for drawing edges
        label_nodes: Whether to include labels
    """
    # Create a graph where nodes are traits
    G = nx.Graph()
    
    # Get all unique traits and add as nodes
    traits = {}  # Map from trait to trait_id
    for agent in agents:
        trait_str = ''.join(agent.trait)
        if trait_str not in traits:
            traits[trait_str] = agent.trait_id
            is_recombined = agent.trait_meta.get('recombined', False)
            G.add_node(agent.trait_id, 
                      trait=trait_str,
                      recombined=is_recombined)
    
    # Add edges based on trait similarity
    for trait1, id1 in traits.items():
        for trait2, id2 in traits.items():
            if id1 != id2:
                # Calculate similarity
                similarity = sum(a == b for a, b in zip(trait1, trait2)) / max(len(trait1), len(trait2))
                if similarity >= threshold:
                    G.add_edge(id1, id2, weight=similarity)
    
    # Create layout
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(12, 10))
    
    # Draw nodes (recombined traits in red, others in blue)
    node_colors = ['red' if G.nodes[n].get('recombined', False) else 'skyblue' 
                 for n in G.nodes()]
    node_sizes = [100 + G.degree(n) * 10 for n in G.nodes()]  # Size by connectivity
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
    
    if label_nodes:
        # Create labels with first 8 chars of trait ID 
        labels = {n: n[:8] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    plt.title(f"Trait Similarity Network (threshold={threshold})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print some network stats
    print(f"Network statistics:")
    print(f"  Nodes (unique traits): {G.number_of_nodes()}")
    print(f"  Edges (similarities): {G.number_of_edges()}")
    try:
        # Try to calculate clustering coefficient
        clustering = nx.average_clustering(G)
        print(f"  Clustering coefficient: {clustering:.3f}")
    except:
        pass
    
    # Count communities
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        print(f"  Communities: {len(communities)}")
        print(f"  Largest community size: {len(communities[0]) if communities else 0}")
    except:
        pass
    
    return G  # Return the graph for further analysis 
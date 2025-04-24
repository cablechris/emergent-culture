#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict
import networkx as nx
from environment import run_simulation
from scipy.spatial.distance import pdist, squareform

def run_recombination_experiment():
    """
    Run simulations with different recombination rates and analyze the results.
    
    Returns:
        Dictionary with results for each recombination rate
    """
    results = {}
    
    # Check if we have saved results
    if os.path.exists('recombination_experiment.pkl'):
        print("Loading saved recombination experiment results...")
        with open('recombination_experiment.pkl', 'rb') as f:
            results = pickle.load(f)
        return results
    
    # Recombination rates to test
    rates = [0.0, 0.05, 0.15]
    
    for rate in rates:
        print(f"Running simulation with recombination rate: {rate}")
        
        # Run simulation with this recombination rate
        interaction_log, agents, modularity, preference_log, imitation_log, reputation_log, lineage_log = run_simulation(
            num_agents=100,
            num_epochs=100,
            use_behaviors=True,
            selective_imitation=True,
            recombination_rate=rate
        )
        
        # Store results
        results[rate] = {
            'agents': agents,
            'imitation_log': imitation_log,
            'lineage_log': lineage_log,
            'modularity': modularity
        }
    
    # Save results
    with open('recombination_experiment.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def count_recombined_traits(imitation_log):
    """
    Count recombined traits per epoch from the imitation log.
    
    Args:
        imitation_log: Dict mapping epochs to imitation events
        
    Returns:
        Dictionary with recombination counts by epoch
    """
    recombination_counts = {}
    
    for epoch, events in imitation_log.items():
        recombination_counts[epoch] = events.get('recombination', 0)
    
    return recombination_counts

def analyze_trait_complexity(agents):
    """
    Analyze the complexity of traits in the final population.
    
    Args:
        agents: List of agents
        
    Returns:
        Dictionary with trait complexity metrics
    """
    metrics = {}
    
    # Get all traits
    all_traits = [''.join(agent.trait) for agent in agents]
    
    # Calculate trait diversity (number of unique traits)
    unique_traits = set(all_traits)
    metrics['trait_diversity'] = len(unique_traits)
    
    # Calculate average trait complexity
    # Here we use character-level entropy as a simple complexity measure
    trait_complexities = []
    for trait in unique_traits:
        # Count frequencies of each character
        char_counts = {}
        for char in trait:
            char_counts[char] = char_counts.get(char, 0) + 1
            
        # Calculate entropy
        entropy = 0
        for count in char_counts.values():
            p = count / len(trait)
            entropy -= p * np.log2(p)
            
        trait_complexities.append(entropy)
    
    metrics['avg_complexity'] = np.mean(trait_complexities)
    metrics['max_complexity'] = np.max(trait_complexities)
    
    # Count recombined traits
    recombined_count = sum(1 for agent in agents if agent.trait_meta.get('recombined', False))
    metrics['recombined_count'] = recombined_count
    metrics['recombined_percent'] = recombined_count / len(agents) * 100
    
    return metrics

def plot_recombination_counts(results):
    """
    Plot the counts of recombination events over time for different rates.
    
    Args:
        results: Dictionary with experiment results
    """
    plt.figure(figsize=(10, 6))
    
    for rate, data in results.items():
        if rate == 0.0:
            continue  # Skip the no-recombination case
            
        imitation_log = data['imitation_log']
        recombination_counts = count_recombined_traits(imitation_log)
        
        epochs = sorted(recombination_counts.keys())
        counts = [recombination_counts[e] for e in epochs]
        
        plt.plot(epochs, counts, marker='o', linestyle='-', 
                 label=f"Rate = {rate}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Recombination Events')
    plt.title('Trait Recombination Events Over Time')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_trait_diversity(results):
    """
    Plot trait diversity for different recombination rates.
    
    Args:
        results: Dictionary with experiment results
    """
    rates = sorted(results.keys())
    diversity = [analyze_trait_complexity(results[rate]['agents'])['trait_diversity'] 
                for rate in rates]
    complexity = [analyze_trait_complexity(results[rate]['agents'])['avg_complexity'] 
                 for rate in rates]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(rates)), diversity, color='skyblue')
    plt.xticks(range(len(rates)), [f"{r}" for r in rates])
    plt.xlabel('Recombination Rate')
    plt.ylabel('Number of Unique Traits')
    plt.title('Trait Diversity by Recombination Rate')
    plt.grid(alpha=0.3, axis='y')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(rates)), complexity, color='salmon')
    plt.xticks(range(len(rates)), [f"{r}" for r in rates])
    plt.xlabel('Recombination Rate')
    plt.ylabel('Average Trait Complexity (Entropy)')
    plt.title('Trait Complexity by Recombination Rate')
    plt.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def plot_modularity_comparison(results):
    """
    Plot modularity over time for different recombination rates.
    
    Args:
        results: Dictionary with experiment results
    """
    plt.figure(figsize=(10, 6))
    
    for rate, data in results.items():
        modularity = data['modularity']
        
        epochs = sorted(modularity.keys())
        values = [modularity[e] for e in epochs]
        
        plt.plot(epochs, values, marker='o' if rate == 0 else None, 
                 linestyle='-', linewidth=2 if rate == 0 else 1.5,
                 label=f"Rate = {rate}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Modularity Score')
    plt.title('Cultural Modularity Over Time by Recombination Rate')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_trait_survival(results):
    """
    Analyze the survival rate of recombinant vs. non-recombinant traits.
    
    Args:
        results: Dictionary with experiment results
        
    Returns:
        Dictionary with survival metrics
    """
    survival_metrics = {}
    
    for rate, data in results.items():
        if rate == 0.0:
            continue  # Skip the baseline case
            
        lineage_log = data['lineage_log']
        
        # Track trait appearances and disappearances
        trait_first_seen = {}  # When trait first appeared
        trait_last_seen = {}   # When trait was last seen
        trait_is_recombined = {}  # Whether trait is recombined
        
        # Process lineage log to identify trait lifespans
        epochs = sorted(lineage_log.keys())
        
        for epoch in epochs:
            epoch_data = lineage_log[epoch]
            current_traits = set()
            
            for agent_data in epoch_data:
                trait_id = agent_data['trait_id']
                current_traits.add(trait_id)
                
                # If this is the first time we've seen this trait, record it
                if trait_id not in trait_first_seen:
                    trait_first_seen[trait_id] = epoch
                
                # Update last seen time for this trait
                trait_last_seen[trait_id] = epoch
                
                # Check agent's metadata to determine if trait is recombined
                agent_id = agent_data['agent_id']
                agent = next((a for a in data['agents'] if a.id == agent_id), None)
                if agent:
                    trait_is_recombined[trait_id] = agent.trait_meta.get('recombined', False)
        
        # Calculate lifespan for each trait
        trait_lifespans = {}
        for trait_id, first_epoch in trait_first_seen.items():
            last_epoch = trait_last_seen[trait_id]
            trait_lifespans[trait_id] = last_epoch - first_epoch + 1  # +1 to count both endpoints
        
        # Separate recombined from non-recombined traits
        recombined_lifespans = [lifespan for trait_id, lifespan in trait_lifespans.items() 
                              if trait_is_recombined.get(trait_id, False)]
        normal_lifespans = [lifespan for trait_id, lifespan in trait_lifespans.items() 
                          if not trait_is_recombined.get(trait_id, False)]
        
        # Calculate survival metrics
        if recombined_lifespans:
            avg_recombined_lifespan = np.mean(recombined_lifespans)
            median_recombined_lifespan = np.median(recombined_lifespans)
            max_recombined_lifespan = np.max(recombined_lifespans)
        else:
            avg_recombined_lifespan = 0
            median_recombined_lifespan = 0
            max_recombined_lifespan = 0
            
        if normal_lifespans:
            avg_normal_lifespan = np.mean(normal_lifespans)
            median_normal_lifespan = np.median(normal_lifespans)
            max_normal_lifespan = np.max(normal_lifespans)
        else:
            avg_normal_lifespan = 0
            median_normal_lifespan = 0
            max_normal_lifespan = 0
        
        # Store results
        survival_metrics[rate] = {
            'recombined_traits': {
                'count': len(recombined_lifespans),
                'avg_lifespan': avg_recombined_lifespan,
                'median_lifespan': median_recombined_lifespan,
                'max_lifespan': max_recombined_lifespan
            },
            'normal_traits': {
                'count': len(normal_lifespans),
                'avg_lifespan': avg_normal_lifespan,
                'median_lifespan': median_normal_lifespan,
                'max_lifespan': max_normal_lifespan
            }
        }
    
    return survival_metrics

def plot_trait_survival_comparison(survival_metrics):
    """
    Plot comparison of survival rates between recombined and normal traits.
    
    Args:
        survival_metrics: Dictionary with survival metrics by recombination rate
    """
    rates = sorted(survival_metrics.keys())
    
    recombined_avg = [survival_metrics[rate]['recombined_traits']['avg_lifespan'] for rate in rates]
    normal_avg = [survival_metrics[rate]['normal_traits']['avg_lifespan'] for rate in rates]
    
    recombined_max = [survival_metrics[rate]['recombined_traits']['max_lifespan'] for rate in rates]
    normal_max = [survival_metrics[rate]['normal_traits']['max_lifespan'] for rate in rates]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Average lifespan
    x = np.arange(len(rates))
    width = 0.35
    
    axes[0].bar(x - width/2, recombined_avg, width, label='Recombined Traits')
    axes[0].bar(x + width/2, normal_avg, width, label='Normal Traits')
    
    axes[0].set_xlabel('Recombination Rate')
    axes[0].set_ylabel('Average Lifespan (Epochs)')
    axes[0].set_title('Average Trait Survival by Type')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(rate) for rate in rates])
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Maximum lifespan
    axes[1].bar(x - width/2, recombined_max, width, label='Recombined Traits')
    axes[1].bar(x + width/2, normal_max, width, label='Normal Traits')
    
    axes[1].set_xlabel('Recombination Rate')
    axes[1].set_ylabel('Maximum Lifespan (Epochs)')
    axes[1].set_title('Maximum Trait Survival by Type')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(rate) for rate in rates])
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_lineage_trees(results):
    """
    Analyze the lineage trees for recombinant traits.
    
    Args:
        results: Dictionary with experiment results
        
    Returns:
        Dictionary with lineage tree metrics
    """
    lineage_metrics = {}
    
    for rate, data in results.items():
        if rate == 0.0:
            continue  # Skip the baseline case
            
        # Build lineage graph from agents data
        G = nx.DiGraph()
        
        # Add all nodes and edges
        for agent in data['agents']:
            trait_id = agent.trait_id
            parent_id = agent.parent_trait_id
            
            # Add node
            G.add_node(trait_id, 
                      recombined=agent.trait_meta.get('recombined', False),
                      epoch=agent.trait_meta.get('epoch', 0))
            
            # Add edge from parent to child
            if parent_id and parent_id != trait_id:  # Avoid self-loops
                G.add_edge(parent_id, trait_id)
                
            # If this is a recombined trait, add second parent too
            if agent.trait_meta.get('recombined', False):
                parent_b_id = agent.trait_meta.get('parent_b_id')
                if parent_b_id:
                    G.add_edge(parent_b_id, trait_id)
        
        # Find recombined traits
        recombined_nodes = [n for n, d in G.nodes(data=True) if d.get('recombined', False)]
        
        # Calculate tree metrics
        tree_sizes = []  # Number of descendants
        tree_depths = []  # Maximum path length from recombinant to leaf
        
        for node in recombined_nodes:
            # Get descendants
            descendants = list(nx.descendants(G, node))
            tree_size = len(descendants)
            
            # Calculate maximum depth
            max_depth = 0
            if descendants:
                # Find leaf nodes (no outgoing edges)
                leaves = [n for n in descendants if G.out_degree(n) == 0]
                
                # Calculate path lengths to all leaves
                for leaf in leaves:
                    try:
                        path_length = len(nx.shortest_path(G, node, leaf)) - 1  # -1 to get edge count
                        max_depth = max(max_depth, path_length)
                    except nx.NetworkXNoPath:
                        # No path found (shouldn't happen in a DAG)
                        pass
                        
            tree_sizes.append(tree_size)
            tree_depths.append(max_depth)
        
        # Calculate metrics
        if tree_sizes:
            avg_tree_size = np.mean(tree_sizes)
            max_tree_size = np.max(tree_sizes)
            avg_tree_depth = np.mean(tree_depths)
            max_tree_depth = np.max(tree_depths)
        else:
            avg_tree_size = 0
            max_tree_size = 0
            avg_tree_depth = 0
            max_tree_depth = 0
        
        # Store results
        lineage_metrics[rate] = {
            'recombined_count': len(recombined_nodes),
            'avg_tree_size': avg_tree_size,
            'max_tree_size': max_tree_size,
            'avg_tree_depth': avg_tree_depth,
            'max_tree_depth': max_tree_depth,
            'lineage_graph': G  # Store the graph for visualization
        }
    
    return lineage_metrics

def plot_lineage_tree_metrics(lineage_metrics):
    """
    Plot metrics about lineage trees for recombinant traits.
    
    Args:
        lineage_metrics: Dictionary with lineage tree metrics by recombination rate
    """
    rates = sorted(lineage_metrics.keys())
    
    tree_sizes = [lineage_metrics[rate]['avg_tree_size'] for rate in rates]
    tree_depths = [lineage_metrics[rate]['avg_tree_depth'] for rate in rates]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Tree sizes
    axes[0].bar(range(len(rates)), tree_sizes, color='skyblue')
    axes[0].set_xticks(range(len(rates)))
    axes[0].set_xticklabels([str(rate) for rate in rates])
    axes[0].set_xlabel('Recombination Rate')
    axes[0].set_ylabel('Average Tree Size (Descendants)')
    axes[0].set_title('Average Lineage Tree Size for Recombinant Traits')
    axes[0].grid(alpha=0.3, axis='y')
    
    # Tree depths
    axes[1].bar(range(len(rates)), tree_depths, color='salmon')
    axes[1].set_xticks(range(len(rates)))
    axes[1].set_xticklabels([str(rate) for rate in rates])
    axes[1].set_xlabel('Recombination Rate')
    axes[1].set_ylabel('Average Tree Depth')
    axes[1].set_title('Average Lineage Tree Depth for Recombinant Traits')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def visualize_largest_lineage_tree(lineage_metrics, rate):
    """
    Visualize the largest lineage tree for a given recombination rate.
    
    Args:
        lineage_metrics: Dictionary with lineage tree metrics by recombination rate
        rate: The recombination rate to visualize
    """
    if rate not in lineage_metrics:
        print(f"No data available for recombination rate {rate}")
        return
    
    G = lineage_metrics[rate]['lineage_graph']
    
    # Find recombined nodes
    recombined_nodes = [n for n, d in G.nodes(data=True) if d.get('recombined', False)]
    
    if not recombined_nodes:
        print(f"No recombined traits found for rate {rate}")
        return
    
    # Find node with most descendants
    most_descendants = 0
    largest_tree_root = None
    
    for node in recombined_nodes:
        descendants = list(nx.descendants(G, node))
        if len(descendants) > most_descendants:
            most_descendants = len(descendants)
            largest_tree_root = node
    
    if not largest_tree_root:
        print(f"No valid tree found for rate {rate}")
        return
    
    # Create subgraph with this root and its descendants
    descendants = list(nx.descendants(G, largest_tree_root))
    nodes = [largest_tree_root] + descendants
    subG = G.subgraph(nodes)
    
    # Create a hierarchical layout
    try:
        pos = nx.nx_agraph.graphviz_layout(subG, prog='dot')
    except:
        # Fallback to spring layout if graphviz not available
        pos = nx.spring_layout(subG, seed=42)
    
    plt.figure(figsize=(12, 10))
    
    # Color nodes: red for recombined, blue for normal
    node_colors = ['red' if G.nodes[n].get('recombined', False) else 'skyblue' for n in subG.nodes()]
    
    # Size nodes by their number of descendants
    node_sizes = [100 + len(list(nx.descendants(subG, n))) * 20 for n in subG.nodes()]
    
    # Draw the network
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(subG, pos, edge_color='gray', 
                          alpha=0.6, width=1.0, arrows=True)
    
    # Add labels for important nodes
    labels = {largest_tree_root: largest_tree_root[:8]}
    
    # Add labels for other recombined nodes
    other_recombineds = [n for n in subG.nodes() 
                        if n != largest_tree_root and G.nodes[n].get('recombined', False)]
    for node in other_recombineds[:5]:  # Label up to 5 other recombined nodes
        labels[node] = node[:8]
    
    nx.draw_networkx_labels(subG, pos, labels=labels, font_size=10)
    
    plt.title(f"Largest Recombinant Trait Lineage Tree (Rate={rate})\nRoot: {largest_tree_root[:8]}, {most_descendants} descendants")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def analyze_subculture_overlap(results):
    """
    Analyze how hybrid/recombined traits create overlap between subcultures.
    
    Args:
        results: Dictionary with experiment results
        
    Returns:
        Dictionary with subculture overlap metrics
    """
    overlap_metrics = {}
    
    for rate, data in results.items():
        agents = data['agents']
        
        # Skip if we don't have enough data
        if len(agents) < 10:
            continue
            
        # Create trait similarity matrix between agents
        n_agents = len(agents)
        similarity_matrix = np.zeros((n_agents, n_agents))
        
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                # Calculate trait similarity
                sim = trait_similarity(agents[i].trait, agents[j].trait)
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric
        
        # Create a graph with edges between similar agents
        G = nx.Graph()
        for i in range(n_agents):
            G.add_node(i, 
                      trait_id=agents[i].trait_id,
                      recombined=agents[i].trait_meta.get('recombined', False))
        
        # Add edges for agents with high trait similarity
        threshold = np.percentile(similarity_matrix[similarity_matrix > 0], 75)  # Top 25% most similar
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                if similarity_matrix[i, j] >= threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
        
        # Detect communities
        communities = list(nx.community.greedy_modularity_communities(G))
        
        # Identify agents with recombined traits
        recombined_agents = [i for i in range(n_agents) if agents[i].trait_meta.get('recombined', False)]
        
        # Count how many recombined traits connect different communities
        bridge_traits = []
        for agent_idx in recombined_agents:
            agent_communities = []
            for neighbor in G.neighbors(agent_idx):
                # Find which community this neighbor belongs to
                for comm_idx, comm in enumerate(communities):
                    if neighbor in comm:
                        agent_communities.append(comm_idx)
                        break
            
            # If this agent connects at least two different communities, it's a bridge
            unique_communities = set(agent_communities)
            if len(unique_communities) >= 2:
                bridge_traits.append(agents[agent_idx].trait_id)
        
        # Calculate metrics
        if recombined_agents:
            bridge_proportion = len(bridge_traits) / len(recombined_agents)
        else:
            bridge_proportion = 0
            
        avg_communities_bridged = 0
        if bridge_traits:
            community_counts = []
            for agent_idx in recombined_agents:
                if agents[agent_idx].trait_id in bridge_traits:
                    agent_communities = set()
                    for neighbor in G.neighbors(agent_idx):
                        for comm_idx, comm in enumerate(communities):
                            if neighbor in comm:
                                agent_communities.add(comm_idx)
                                break
                    community_counts.append(len(agent_communities))
            
            avg_communities_bridged = np.mean(community_counts) if community_counts else 0
        
        # Store results
        overlap_metrics[rate] = {
            'num_communities': len(communities),
            'num_recombined': len(recombined_agents),
            'num_bridge_traits': len(bridge_traits),
            'bridge_proportion': bridge_proportion,
            'avg_communities_bridged': avg_communities_bridged,
            'agent_graph': G,
            'communities': communities
        }
    
    return overlap_metrics

def trait_similarity(trait1, trait2):
    """Calculate similarity between two traits"""
    return sum(a == b for a, b in zip(trait1, trait2)) / max(len(trait1), len(trait2))

def plot_subculture_overlap(overlap_metrics):
    """
    Plot metrics about how recombined traits bridge subcultures.
    
    Args:
        overlap_metrics: Dictionary with subculture overlap metrics by recombination rate
    """
    rates = sorted(overlap_metrics.keys())
    
    bridge_proportions = [overlap_metrics[rate]['bridge_proportion'] * 100 for rate in rates]
    avg_bridged = [overlap_metrics[rate]['avg_communities_bridged'] for rate in rates]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bridge proportions
    axes[0].bar(range(len(rates)), bridge_proportions, color='skyblue')
    axes[0].set_xticks(range(len(rates)))
    axes[0].set_xticklabels([str(rate) for rate in rates])
    axes[0].set_xlabel('Recombination Rate')
    axes[0].set_ylabel('Bridge Traits (%)')
    axes[0].set_title('Percentage of Recombined Traits\nThat Bridge Communities')
    axes[0].grid(alpha=0.3, axis='y')
    
    # Average communities bridged
    axes[1].bar(range(len(rates)), avg_bridged, color='salmon')
    axes[1].set_xticks(range(len(rates)))
    axes[1].set_xticklabels([str(rate) for rate in rates])
    axes[1].set_xlabel('Recombination Rate')
    axes[1].set_ylabel('Average Communities Bridged')
    axes[1].set_title('Average Number of Communities\nConnected by Bridge Traits')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def visualize_subculture_network(overlap_metrics, rate):
    """
    Visualize the network of agents with communities and bridge traits highlighted.
    
    Args:
        overlap_metrics: Dictionary with subculture overlap metrics by recombination rate
        rate: The recombination rate to visualize
    """
    if rate not in overlap_metrics:
        print(f"No data available for recombination rate {rate}")
        return
    
    data = overlap_metrics[rate]
    G = data['agent_graph']
    communities = data['communities']
    
    # Create position layout with communities
    try:
        pos = nx.spring_layout(G, k=0.3, seed=42)
    except:
        pos = nx.random_layout(G)
    
    plt.figure(figsize=(12, 10))
    
    # Assign colors to communities
    community_colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))
    
    # Draw nodes by community
    for i, comm in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=list(comm), 
                              node_color=[community_colors[i]] * len(comm),
                              node_size=100, 
                              alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
    
    # Highlight recombined traits that bridge communities
    bridge_nodes = []
    for node in G.nodes():
        if G.nodes[node].get('recombined', False):
            # Count different communities in neighbors
            neighbor_communities = set()
            for neighbor in G.neighbors(node):
                for comm_idx, comm in enumerate(communities):
                    if neighbor in comm:
                        neighbor_communities.add(comm_idx)
                        break
                        
            if len(neighbor_communities) >= 2:
                bridge_nodes.append(node)
    
    # Draw bridge nodes with a different color and larger size
    if bridge_nodes:
        nx.draw_networkx_nodes(G, pos,
                              nodelist=bridge_nodes,
                              node_color='red',
                              node_size=300,
                              alpha=1.0)
    
    plt.title(f"Agent Similarity Network with Communities (Rate={rate})\n"
              f"{len(communities)} communities, {len(bridge_nodes)} bridge traits")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """Run the recombination analysis"""
    
    # Run experiment
    results = run_recombination_experiment()
    
    # Summarize results
    print("\nTrait Complexity Analysis:")
    for rate, data in results.items():
        metrics = analyze_trait_complexity(data['agents'])
        print(f"\nRecombination Rate: {rate}")
        print(f"  Trait Diversity: {metrics['trait_diversity']} unique traits")
        print(f"  Avg Complexity: {metrics['avg_complexity']:.3f}")
        print(f"  Recombined Traits: {metrics['recombined_count']} ({metrics['recombined_percent']:.1f}%)")
    
    # Plot figures
    print("\n1. Recombination Events Over Time")
    plot_recombination_counts(results)
    
    print("\n2. Trait Diversity and Complexity")
    plot_trait_diversity(results)
    
    print("\n3. Cultural Modularity Comparison")
    plot_modularity_comparison(results)
    
    # Run trait survival analysis
    print("\n4. Trait Survival Analysis")
    survival_metrics = analyze_trait_survival(results)
    plot_trait_survival_comparison(survival_metrics)
    
    # Run lineage tree analysis
    print("\n5. Lineage Tree Analysis for Recombinants")
    lineage_metrics = analyze_lineage_trees(results)
    plot_lineage_tree_metrics(lineage_metrics)
    
    # Visualize a sample lineage tree for a non-zero recombination rate
    for rate in sorted(lineage_metrics.keys()):
        print(f"\nVisualizing largest lineage tree for rate {rate}")
        visualize_largest_lineage_tree(lineage_metrics, rate)
        break  # Just show one example
    
    # Run subculture overlap analysis
    print("\n6. Subculture Overlap via Hybrid Traits")
    overlap_metrics = analyze_subculture_overlap(results)
    plot_subculture_overlap(overlap_metrics)
    
    # Visualize an example network
    for rate in sorted(overlap_metrics.keys()):
        print(f"\nVisualizing agent network with communities for rate {rate}")
        visualize_subculture_network(overlap_metrics, rate)
        break  # Just show one example

if __name__ == "__main__":
    main() 
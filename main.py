#!/usr/bin/env python3

from environment import run_simulation
import matplotlib.pyplot as plt
from collections import Counter
import math
import networkx as nx
import numpy as np
from analysis_utils import compute_modularity, build_agent_graph, track_cluster_preferences
try:
    import community as community_louvain
except ImportError:
    import community.community_louvain as community_louvain

def shannon_entropy(traits):
    counts = Counter(traits)
    total = sum(counts.values())
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

def plot_entropy(log):
    traits_by_epoch = [[''.join(t[2]) for t in log[epoch]] for epoch in sorted(log)]
    entropies = [shannon_entropy(traits) for traits in traits_by_epoch]
    plt.plot(entropies)
    plt.title('Trait Entropy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Shannon Entropy')
    plt.grid(True)
    plt.show()

def plot_top_traits(log, top_n=5):
    trait_freqs = {}
    for epoch in sorted(log):
        traits = [t[2] for t in log[epoch]]
        freqs = Counter(traits)
        trait_freqs[epoch] = freqs

    all_traits = Counter()
    for freqs in trait_freqs.values():
        all_traits.update(freqs)

    top_traits = [trait for trait, _ in all_traits.most_common(top_n)]

    for trait in top_traits:
        history = []
        for epoch in sorted(trait_freqs):
            history.append(trait_freqs[epoch].get(trait, 0))
        plt.plot(history, label=trait)

    plt.title(f'Top {top_n} Traits Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def trait_similarity(trait1, trait2):
    return sum(1 for a, b in zip(trait1, trait2) if a == b)

def build_trait_graph(agents, similarity_threshold=6):
    G = nx.Graph()
    for agent in agents:
        G.add_node(agent.id, trait=''.join(agent.trait))

    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            similarity = trait_similarity(a1.trait, a2.trait)
            if similarity >= similarity_threshold:
                G.add_edge(a1.id, a2.id)

    return G

def plot_subcultures(agents):
    G = build_trait_graph(agents)
    components = list(nx.connected_components(G))
    color_map = {}
    for i, group in enumerate(components):
        for node in group:
            color_map[node] = i

    node_colors = [color_map[n] for n in G.nodes()]
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.rainbow)
    plt.title('Agent Trait-Based Subcultures')
    plt.show()

def plot_average_cost(log):
    avg_costs = []
    for epoch in sorted(log):
        costs = [entry[3] for entry in log[epoch]]  # index 3 is cost
        avg = sum(costs) / len(costs)
        avg_costs.append(avg)
    
    plt.plot(avg_costs)
    plt.title("Average Signal Cost Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Average Cost")
    plt.grid(True)
    plt.show()

def plot_cost_vs_trait_frequency(log):
    from collections import defaultdict
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    trait_counts = defaultdict(int)
    trait_costs = {}

    for epoch in log:
        for _, _, trait, cost in log[epoch]:
            trait_counts[trait] += 1
            trait_costs[trait] = cost  # last seen cost

    costs = []
    freqs = []
    for trait in trait_counts:
        costs.append(trait_costs[trait])
        freqs.append(trait_counts[trait])

    # Scatter plot
    plt.scatter(costs, freqs, alpha=0.6)

    # Trend line (optional)
    z = np.polyfit(costs, freqs, 1)
    p = np.poly1d(z)
    plt.plot(sorted(costs), p(sorted(costs)), linestyle='--', color='red')

    # Pearson correlation
    r, _ = pearsonr(costs, freqs)
    plt.title(f"Cost vs Trait Frequency\n(Pearson r = {r:.2f})")
    plt.xlabel("Trait Cost")
    plt.ylabel("Imitation Count")
    plt.grid(True)
    plt.show()

def plot_cost_vs_lifespan(log):
    from collections import defaultdict
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr
    import numpy as np

    trait_epochs = defaultdict(list)
    trait_costs = {}

    for epoch in log:
        for _, _, trait, cost in log[epoch]:
            trait_epochs[trait].append(epoch)
            trait_costs[trait] = cost

    lifespans = {
        trait: max(epochs) - min(epochs) + 1
        for trait, epochs in trait_epochs.items()
    }

    costs = [trait_costs[t] for t in lifespans]
    span = [lifespans[t] for t in lifespans]

    # Correlation
    r_pearson, _ = pearsonr(costs, span)
    r_spearman, _ = spearmanr(costs, span)

    # Plot
    plt.scatter(costs, span, alpha=0.6)
    z = np.polyfit(costs, span, 1)
    p = np.poly1d(z)
    plt.plot(sorted(costs), p(sorted(costs)), 'r--')
    plt.title(f"Cost vs Trait Lifespan\nPearson r = {r_pearson:.2f}, Spearman ρ = {r_spearman:.2f}")
    plt.xlabel("Trait Cost")
    plt.ylabel("Lifespan (epochs)")
    plt.grid(True)
    plt.show()

def plot_trait_survival_curve(log):
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # Record trait lifespan ranges
    trait_birth = {}
    trait_death = {}

    for epoch in sorted(log):
        for _, _, trait, _ in log[epoch]:
            if trait not in trait_birth:
                trait_birth[trait] = epoch
            trait_death[trait] = epoch

    # Compute lifespan duration per trait
    trait_lifespans = {t: (trait_birth[t], trait_death[t]) for t in trait_birth}

    # Get number of traits still alive at each epoch
    max_epoch = max(trait_death.values())
    survival_counts = []

    for epoch in range(max_epoch + 1):
        alive = sum(1 for start, end in trait_lifespans.values() if start <= epoch <= end)
        survival_counts.append(alive)

    total = len(trait_lifespans)
    survival_fraction = [count / total for count in survival_counts]

    # Plot
    plt.plot(survival_fraction)
    plt.title("Trait Survival Curve (Kaplan-Meier Style)")
    plt.xlabel("Epoch")
    plt.ylabel("Fraction of Traits Still Alive")
    plt.grid(True)
    plt.show()

def plot_modularity(modularity_log):
    import matplotlib.pyplot as plt

    epochs = sorted(modularity_log.keys())
    values = [modularity_log[e] for e in epochs]

    plt.plot(epochs, values)
    plt.title("Subcultural Modularity Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Modularity Score")
    plt.grid(True)
    plt.show()

def plot_colored_subcultures(agents, similarity_threshold=6):
    import matplotlib.pyplot as plt
    import networkx as nx
    import community as community_louvain

    def trait_similarity(t1, t2):
        return sum(a == b for a, b in zip(t1, t2))

    # Build graph
    G = nx.Graph()
    for agent in agents:
        G.add_node(agent.id)

    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            sim = trait_similarity(a1.trait, a2.trait)
            if sim >= similarity_threshold:
                G.add_edge(a1.id, a2.id)

    if len(G.edges) == 0:
        print("No edges — no visual clustering possible.")
        return

    # Detect communities
    partition = community_louvain.best_partition(G)

    # Assign colors by community
    color_map = [partition[node] for node in G.nodes()]
    pos = nx.spring_layout(G, seed=42)

    # Draw
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, node_color=color_map, with_labels=False, node_size=80, cmap=plt.cm.tab20)
    plt.title("Trait-Based Subcultures (Community Colored)")
    plt.show()

def plot_trait_behavior_complexity(agents):
    """
    Create a scatter plot showing the relationship between trait and behavior complexity.
    Complexity is measured by the number of unique symbols.
    """
    import matplotlib.pyplot as plt
    
    trait_complexity = [len(set(agent.trait)) for agent in agents]
    behavior_complexity = [len(set(agent.behavior)) for agent in agents]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(trait_complexity, behavior_complexity, alpha=0.6)
    
    plt.title("Trait vs Behavior Complexity")
    plt.xlabel("Trait Complexity (unique symbols)")
    plt.ylabel("Behavior Complexity (unique symbols)")
    plt.grid(True)
    
    # Calculate and display correlation
    from scipy.stats import pearsonr
    corr, _ = pearsonr(trait_complexity, behavior_complexity)
    plt.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction')
    
    plt.show()

def plot_preference_distribution(agents):
    """
    Create a histogram showing the distribution of agent preferences 
    for traits vs behaviors.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract preference for traits (index 0 of preference vector)
    trait_preferences = [agent.preference_vector[0] for agent in agents]
    
    plt.figure(figsize=(8, 6))
    plt.hist(trait_preferences, bins=20, alpha=0.7)
    plt.axvline(np.mean(trait_preferences), color='r', linestyle='dashed', linewidth=1)
    
    plt.title("Distribution of Agent Preferences for Traits vs Behaviors")
    plt.xlabel("Preference for Traits (vs Behaviors)")
    plt.ylabel("Number of Agents")
    plt.annotate(f"Mean: {np.mean(trait_preferences):.2f}", 
                xy=(np.mean(trait_preferences), plt.ylim()[1]*0.9))
    plt.grid(True)
    plt.show()

def plot_dual_signal_network(agents, similarity_threshold=0.5):
    """
    Create a network visualization showing connections between agents
    based on both trait and behavior similarity.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from agent import trait_similarity, behavior_similarity
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes
    for a in agents:
        G.add_node(a.id)
    
    # Calculate max possible similarity for normalization
    max_trait_sim = len(agents[0].trait)
    max_behavior_sim = len(agents[0].behavior)
    
    # Add edges based on weighted trait and behavior similarity
    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            # Normalize similarities to 0-1 range
            trait_sim = trait_similarity(a1.trait, a2.trait) / max_trait_sim
            behavior_sim = behavior_similarity(a1.behavior, a2.behavior) / max_behavior_sim
            
            # Calculate combined similarity
            # Weight by average preference across the two agents
            a1_trait_pref = a1.preference_vector[0]
            a2_trait_pref = a2.preference_vector[0]
            a1_behavior_pref = a1.preference_vector[1]
            a2_behavior_pref = a2.preference_vector[1]
            
            avg_trait_pref = (a1_trait_pref + a2_trait_pref) / 2
            avg_behavior_pref = (a1_behavior_pref + a2_behavior_pref) / 2
            
            combined_sim = (trait_sim * avg_trait_pref) + (behavior_sim * avg_behavior_pref)
            
            if combined_sim >= similarity_threshold:
                G.add_edge(a1.id, a2.id, weight=combined_sim)
    
    # Get node positions using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the network
    plt.figure(figsize=(10, 8))
    
    # Use community detection to color nodes
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        cmap = plt.cm.rainbow
        node_colors = [partition[node] for node in G.nodes()]
    except:
        # Default coloring if community detection fails
        node_colors = 'skyblue'
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.rainbow, 
                          node_size=100, alpha=0.8)
    
    # Draw edges with width based on similarity
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3)
    
    plt.title("Agent Network: Combined Trait and Behavior Similarity")
    plt.axis('off')
    
    # Add a note about the threshold
    plt.annotate(f"Connection threshold: {similarity_threshold}", 
                xy=(0.02, 0.02), xycoords='axes fraction')
    
    # Show modularity
    if 'community_louvain' in locals():
        modularity = community_louvain.modularity(partition, G)
        plt.annotate(f"Modularity: {modularity:.2f}", 
                    xy=(0.02, 0.06), xycoords='axes fraction')
    
    plt.show()

def plot_channel_entropy(log):
    import matplotlib.pyplot as plt
    import math
    from collections import Counter

    def shannon_entropy(tokens):
        counts = Counter(tokens)
        total = sum(counts.values())
        return -sum((count / total) * math.log2(count / total) for count in counts.values())

    trait_entropy = []
    behavior_entropy = []

    for epoch in sorted(log.keys()):
        traits = [entry[2] for entry in log[epoch]]  # trait
        behaviors = [entry[3] for entry in log[epoch]]  # behavior
        trait_entropy.append(shannon_entropy(traits))
        behavior_entropy.append(shannon_entropy(behaviors))

    plt.plot(trait_entropy, label="Trait Entropy")
    plt.plot(behavior_entropy, label="Behavior Entropy")
    plt.title("Channel-Specific Entropy Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Shannon Entropy")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_preference_drift(preference_log):
    import matplotlib.pyplot as plt
    import numpy as np

    epochs = sorted(preference_log.keys())
    avg_trait_prefs = [np.mean([p[0] for p in preference_log[ep]]) for ep in epochs]

    plt.plot(epochs, avg_trait_prefs)
    plt.axhline(0.5, linestyle="--", color="red", alpha=0.6)
    plt.title("Preference Drift Over Time (Trait Preference)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Trait Preference")
    plt.grid(True)
    plt.show()

def plot_cluster_preferences(agents):
    """
    Create a bar chart showing how trait preferences vary across different clusters.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Build a graph based on trait and behavior similarity
    G = build_agent_graph(agents, use_behaviors=True, similarity_threshold=0.5)
    
    # Get average preferences by cluster
    cluster_prefs = track_cluster_preferences(agents, G)
    
    # Sort clusters by average preference for easier interpretation
    sorted_clusters = sorted(cluster_prefs.items(), key=lambda x: x[1])
    cluster_ids, preferences = zip(*sorted_clusters)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(cluster_ids)), preferences, color='skyblue')
    
    # Add horizontal line at 0.5 to indicate equal preference
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    
    # Add labels and formatting
    plt.xlabel('Community Cluster')
    plt.ylabel('Average Trait Preference')
    plt.title('Trait Preference by Community Cluster')
    plt.xticks(range(len(cluster_ids)), [f"Cluster {c}" for c in cluster_ids])
    plt.ylim(0, 1)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{preferences[i]:.2f}', ha='center', va='bottom')
    
    # Add annotation explaining the meaning
    plt.annotate('Values > 0.5: Prefer traits over behaviors', xy=(0.05, 0.95), xycoords='axes fraction')
    plt.annotate('Values < 0.5: Prefer behaviors over traits', xy=(0.05, 0.9), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.show()

def plot_imitation_patterns(imitation_log):
    """
    Visualize the patterns of different types of imitation events over time.
    
    Args:
        imitation_log: Dictionary mapping epochs to counts of different imitation types
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    
    # Categories to track
    categories = ['T+B', 'T-only', 'B-only', 'None']
    colors = ['green', 'blue', 'orange', 'red']
    
    # Convert log into a format easier to plot
    epochs = sorted(imitation_log.keys())
    data = defaultdict(list)
    
    for epoch in epochs:
        for cat in categories:
            data[cat].append(imitation_log[epoch].get(cat, 0))
    
    # Create a stacked area chart
    plt.figure(figsize=(10, 6))
    
    # Ensure all categories have values for all epochs
    for cat in categories:
        if cat not in data:
            data[cat] = [0] * len(epochs)
    
    # Normalize to show proportions
    if len(epochs) > 0:
        sums = [sum(data[cat][i] for cat in categories) for i in range(len(epochs))]
        for cat in categories:
            data[cat] = [data[cat][i] / max(1, sums[i]) for i in range(len(epochs))]
    
    # Stack areas
    plt.stackplot(epochs, 
                 [data['T+B'], data['T-only'], data['B-only'], data['None']], 
                 labels=categories,
                 colors=colors,
                 alpha=0.7)
    
    plt.title('Proportion of Imitation Types Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Proportion of Imitations')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.show()

if __name__ == '__main__':
    log, final_agents, modularity_by_epoch, preference_log, imitation_log = run_simulation(
        use_behaviors=True, 
        selective_imitation=True
    )
    
    # Calculate and display modularity
    mod_value = compute_modularity(final_agents)
    print(f"Network Modularity: {mod_value:.4f}")
    
    # Original visualizations
    plot_entropy(log)
    plot_top_traits(log)
    plot_subcultures(final_agents)
    plot_colored_subcultures(final_agents)
    plot_average_cost(log)
    plot_cost_vs_trait_frequency(log)
    plot_cost_vs_lifespan(log)
    plot_trait_survival_curve(log)
    plot_modularity(modularity_by_epoch)
    
    # New visualizations for trait-behavior relationships
    plot_trait_behavior_complexity(final_agents)
    plot_preference_distribution(final_agents)
    plot_dual_signal_network(final_agents)
    plot_channel_entropy(log)
    plot_preference_drift(preference_log)
    plot_cluster_preferences(final_agents)
    plot_imitation_patterns(imitation_log)


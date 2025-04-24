#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict
import networkx as nx

def encode_trait(trait):
    """
    Convert a trait to a vector representation.
    This is a simple example - in a real implementation you might
    use a more sophisticated embedding.
    
    Args:
        trait: The trait to encode
        
    Returns:
        A vector representation of the trait
    """
    # Simple hash-based encoding
    # Convert trait to a number using hash
    hash_val = hash(str(trait)) % 100000
    
    # Create a small vector based on the hash
    # This is a very simple approach - you can make this more sophisticated
    vec_size = 5
    vec = np.zeros(vec_size)
    for i in range(vec_size):
        vec[i] = ((hash_val >> (i * 4)) & 0xF) / 15.0  # Normalize to [0,1]
        
    return vec

def encode_behavior(behavior):
    """
    Convert a behavior to a vector representation.
    
    Args:
        behavior: The behavior to encode
        
    Returns:
        A vector representation of the behavior
    """
    # Similar approach to trait encoding
    hash_val = hash(str(behavior)) % 100000
    
    vec_size = 5
    vec = np.zeros(vec_size)
    for i in range(vec_size):
        vec[i] = ((hash_val >> (i * 4)) & 0xF) / 15.0
        
    return vec

def update_preference_vector(agent, observed_trait, observed_behavior, learning_rate=0.1):
    """
    Update an agent's preference vector based on observed trait and behavior.
    
    Args:
        agent: The agent whose preferences should be updated
        observed_trait: The trait observed
        observed_behavior: The behavior observed 
        learning_rate: How quickly to adapt to the new observation
    """
    # Check the shape of agent's preference vector
    if len(agent.preference_vector) != 2:
        print(f"Warning: Agent preference vector has unexpected shape {agent.preference_vector.shape}")
        return
        
    # For Agent class in our environment, preference_vector has only 2 components
    # [trait_preference, behavior_preference]
    # We don't need to encode the traits, just update the weights directly
    
    # Direct approach for Agent class: just slightly shift preference 
    # toward the type of signal that was observed
    trait_weight = agent.preference_vector[0]
    behavior_weight = agent.preference_vector[1]
    
    # Adjust weights based on observation - increase weight for the observed type
    if learning_rate > 0:
        # Slightly increase preference for observed trait/behavior type
        new_trait_weight = trait_weight * (1 - learning_rate) + learning_rate
        new_behavior_weight = behavior_weight * (1 - learning_rate) + learning_rate
        
        # Normalize to ensure sum = 1
        total = new_trait_weight + new_behavior_weight
        agent.preference_vector[0] = new_trait_weight / total
        agent.preference_vector[1] = new_behavior_weight / total

def initialize_preference_vector(dim=10):
    """Create an initial random preference vector"""
    return np.random.random(dim)

def calculate_preference_distance(vec1, vec2):
    """Calculate distance between two preference vectors"""
    return np.linalg.norm(vec1 - vec2)

def run_preference_simulation(num_agents=100, num_epochs=50, 
                             learning_rate=0.1, 
                             interaction_probability=0.3):
    """
    Run a simulation of preference evolution.
    
    Args:
        num_agents: Number of agents in the simulation
        num_epochs: Number of epochs to run
        learning_rate: Rate at which preferences update
        interaction_probability: Chance of agents interacting
        
    Returns:
        Dictionary with simulation data
    """
    class SimAgent:
        def __init__(self, id):
            self.id = id
            self.preference_vector = initialize_preference_vector()
            self.trait = f"trait_{id % 10}"  # Simple initial traits
            self.behavior = f"behavior_{id % 5}"  # Simple initial behaviors
    
    # Create agents
    agents = [SimAgent(i) for i in range(num_agents)]
    
    # Store data for analysis
    history = defaultdict(list)
    
    # Run simulation
    for epoch in range(num_epochs):
        # Save current state
        agent_states = []
        for agent in agents:
            agent_states.append({
                'id': agent.id,
                'preference_vector': agent.preference_vector.copy(),
                'trait': agent.trait,
                'behavior': agent.behavior
            })
        history[epoch] = agent_states
        
        # Simulate interactions
        for i, agent in enumerate(agents):
            # Each agent may interact with other random agents
            for _ in range(5):  # Each agent has 5 potential interactions
                if np.random.random() < interaction_probability:
                    # Select random other agent
                    other_idx = np.random.randint(0, num_agents)
                    if other_idx != i:
                        other = agents[other_idx]
                        
                        # Agent observes other's trait and behavior
                        update_preference_vector(
                            agent, 
                            other.trait, 
                            other.behavior, 
                            learning_rate
                        )
                        
                        # Simple trait/behavior adoption based on similarity
                        if np.random.random() < 0.2:  # 20% chance to adopt trait
                            agent.trait = other.trait
                        
                        if np.random.random() < 0.1:  # 10% chance to adopt behavior
                            agent.behavior = other.behavior
    
    return history

def visualize_preference_clustering(history, epoch=None):
    """
    Visualize how agent preferences cluster over time.
    
    Args:
        history: Simulation history dictionary
        epoch: Which epoch to visualize (if None, visualize final)
    """
    if epoch is None:
        epoch = max(history.keys())
    
    agent_states = history[epoch]
    
    # Extract preference vectors
    preference_vecs = np.array([state['preference_vector'] for state in agent_states])
    
    # Use PCA to reduce to 2D for visualization
    from sklearn.decomposition import PCA
    try:
        pca = PCA(n_components=2)
        reduced_vecs = pca.fit_transform(preference_vecs)
        
        # Create a graph of agents based on preference similarity
        G = nx.Graph()
        
        # Add nodes
        for i, state in enumerate(agent_states):
            G.add_node(i, 
                      trait=state['trait'],
                      behavior=state['behavior'],
                      position=reduced_vecs[i])
        
        # Add edges between similar agents
        threshold = np.percentile([calculate_preference_distance(
            preference_vecs[i], preference_vecs[j])
            for i in range(len(agent_states))
            for j in range(i+1, len(agent_states))], 10)  # Connect closest 10%
        
        for i in range(len(agent_states)):
            for j in range(i+1, len(agent_states)):
                dist = calculate_preference_distance(preference_vecs[i], preference_vecs[j])
                if dist < threshold:
                    G.add_edge(i, j, weight=1.0 - dist/threshold)
        
        # Get positions from PCA
        pos = {i: reduced_vecs[i] for i in range(len(agent_states))}
        
        # Get trait colors - assign colors to unique traits
        traits = [state['trait'] for state in agent_states]
        unique_traits = list(set(traits))
        
        trait_colors = {t: plt.cm.tab20(i/len(unique_traits)) 
                      for i, t in enumerate(unique_traits)}
        node_colors = [trait_colors[G.nodes[n]['trait']] for n in G.nodes]
        
        # Visualize
        plt.figure(figsize=(12, 10))
        
        # Draw edges with transparency based on weight
        for (u, v, d) in G.edges(data=True):
            plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                    color='gray', alpha=d['weight']*0.8, linewidth=1)
        
        # Draw nodes
        plt.scatter([pos[i][0] for i in G.nodes], 
                   [pos[i][1] for i in G.nodes],
                   c=node_colors, s=80, alpha=0.8)
        
        # Show legend for traits
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=trait_colors[t], 
                                 markersize=10, label=t)
                          for t in unique_traits]
        plt.legend(handles=legend_elements, title="Traits", loc="upper right")
        
        plt.title(f"Agent Preference Clustering at Epoch {epoch}")
        plt.xlabel(f"Preference Dimension 1 (PCA)")
        plt.ylabel(f"Preference Dimension 2 (PCA)")
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("This visualization requires scikit-learn for PCA. Please install with: pip install scikit-learn")

def visualize_preference_evolution(history):
    """
    Visualize how preferences evolve over time.
    
    Args:
        history: Simulation history dictionary
    """
    epochs = sorted(history.keys())
    
    # Calculate average preference vector per epoch
    avg_preferences = np.array([
        np.mean([state['preference_vector'] for state in history[epoch]], axis=0)
        for epoch in epochs
    ])
    
    # Calculate diversity (average pairwise distance) per epoch
    diversity = []
    for epoch in epochs:
        preference_vecs = [state['preference_vector'] for state in history[epoch]]
        
        # Calculate average pairwise distance
        n = len(preference_vecs)
        if n > 1:
            total_dist = sum(
                calculate_preference_distance(preference_vecs[i], preference_vecs[j])
                for i in range(n) for j in range(i+1, n)
            )
            avg_dist = total_dist / (n * (n-1) / 2)  # Number of pairs
        else:
            avg_dist = 0
        diversity.append(avg_dist)
    
    # Calculate trait counts per epoch
    trait_diversity = []
    for epoch in epochs:
        traits = [state['trait'] for state in history[epoch]]
        unique_traits = set(traits)
        trait_diversity.append(len(unique_traits))
    
    # Create multi-panel visualization
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot average preference components over time
    for i in range(avg_preferences.shape[1]):
        axes[0].plot(epochs, avg_preferences[:, i], label=f"Dim {i+1}")
    
    axes[0].set_title("Average Preference Vector Components")
    axes[0].set_ylabel("Value")
    axes[0].grid(alpha=0.3)
    if avg_preferences.shape[1] <= 10:  # Only show legend if not too crowded
        axes[0].legend(loc="upper right")
    
    # Plot preference diversity over time
    axes[1].plot(epochs, diversity, 'g-o')
    axes[1].set_title("Preference Diversity (Average Pairwise Distance)")
    axes[1].set_ylabel("Distance")
    axes[1].grid(alpha=0.3)
    
    # Plot trait diversity over time
    axes[2].plot(epochs, trait_diversity, 'r-s')
    axes[2].set_title("Number of Unique Traits")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Count")
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Run preference learning simulation and visualize results"""
    if os.path.exists('preference_sim_data.pkl'):
        print("Loading existing preference simulation data...")
        with open('preference_sim_data.pkl', 'rb') as f:
            history = pickle.load(f)
    else:
        print("Running preference learning simulation...")
        history = run_preference_simulation(
            num_agents=100, 
            num_epochs=30,
            learning_rate=0.1
        )
        
        # Save data
        with open('preference_sim_data.pkl', 'wb') as f:
            pickle.dump(history, f)
    
    # Visualize results
    print("\n1. Preference Clustering")
    visualize_preference_clustering(history)
    
    print("\n2. Preference Evolution")
    visualize_preference_evolution(history)

if __name__ == "__main__":
    main() 
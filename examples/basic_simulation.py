#!/usr/bin/env python3

"""
Basic simulation example for emergent culture framework.
This script demonstrates how to run a simple simulation and visualize the results.
"""

import sys
import os
import pickle
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment import run_simulation
from analysis_utils import compute_modularity
from plotting import plot_trait_frequency_over_time

def run_basic_simulation(num_agents=50, num_epochs=50, save_results=True):
    """
    Run a basic simulation of cultural evolution.
    
    Args:
        num_agents: Number of agents in the simulation
        num_epochs: Number of simulation epochs
        save_results: Whether to save results to file
        
    Returns:
        Tuple of (interaction_log, agents, modularity_metrics, etc.)
    """
    print(f"Running basic simulation with {num_agents} agents for {num_epochs} epochs...")
    
    # Run the simulation
    interaction_log, agents, modularity, preference_log, imitation_log, reputation_log, lineage_log = run_simulation(
        num_agents=num_agents,
        num_epochs=num_epochs,
        use_behaviors=True,
        selective_imitation=True
    )
    
    # Save results if requested
    if save_results:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, 'basic_simulation.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump((interaction_log, agents, modularity, preference_log, imitation_log), f)
        print(f"Results saved to {output_file}")
    
    return interaction_log, agents, modularity, preference_log, imitation_log, reputation_log, lineage_log

def visualize_basic_results(interaction_log, modularity):
    """Simple visualization of basic simulation results"""
    
    # Plot the distribution of traits
    plt.figure(figsize=(10, 6))
    plot_trait_frequency_over_time(interaction_log, top_n=5)
    
    # Plot modularity over time
    plt.figure(figsize=(10, 6))
    epochs = sorted(modularity.keys())
    modularity_values = [modularity[e] for e in epochs]
    
    plt.plot(epochs, modularity_values, 'b-o')
    plt.title('Cultural Modularity Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Modularity Score')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nSimulation Summary:")
    print(f"Final modularity: {modularity_values[-1]:.4f}")
    print(f"Number of distinct traits: {len(set([entry[2] for entries in interaction_log.values() for entry in entries]))}")

if __name__ == "__main__":
    # Run a basic simulation
    interaction_log, agents, modularity, preference_log, imitation_log, reputation_log, lineage_log = run_basic_simulation()
    
    # Visualize the results
    visualize_basic_results(interaction_log, modularity) 
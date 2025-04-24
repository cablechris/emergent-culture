#!/usr/bin/env python3

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from environment import run_simulation

def plot_reputation_distribution(log):
    """
    Visualize the distribution of agent reputations at the final epoch.
    
    Args:
        log: Dictionary mapping epochs to lists of agent reputation values
    """
    last_epoch = max(log.keys())
    reps = log[last_epoch]

    plt.hist(reps, bins=20)
    plt.title("Reputation Distribution at Final Epoch")
    plt.xlabel("Reputation")
    plt.ylabel("Agent Count")
    plt.grid(True)
    plt.show()
    
def plot_reputation_over_time(log):
    """
    Visualize how mean and max reputation evolve over time.
    
    Args:
        log: Dictionary mapping epochs to lists of agent reputation values
    """
    epochs = sorted(log.keys())
    mean_reps = [np.mean(log[epoch]) for epoch in epochs]
    max_reps = [np.max(log[epoch]) for epoch in epochs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_reps, 'b-', label='Mean Reputation')
    plt.plot(epochs, max_reps, 'r-', label='Max Reputation')
    plt.title('Reputation Evolution Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Reputation')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_top_agents_reputation(log, top_n=5):
    """
    Track the reputation of the top N agents over time.
    
    Args:
        log: Dictionary mapping epochs to lists of agent reputation values
        top_n: Number of top agents to track
    """
    last_epoch = max(log.keys())
    # Get IDs of top agents in the final epoch
    top_agents = np.argsort(log[last_epoch])[-top_n:][::-1]
    
    plt.figure(figsize=(10, 6))
    epochs = sorted(log.keys())
    
    for agent_id in top_agents:
        agent_rep = [log[epoch][agent_id] if agent_id < len(log[epoch]) else 0 for epoch in epochs]
        plt.plot(epochs, agent_rep, label=f'Agent {agent_id}')
    
    plt.title(f'Top {top_n} Agents by Final Reputation')
    plt.xlabel('Epoch')
    plt.ylabel('Reputation')
    plt.legend()
    plt.grid(True)
    plt.show()

# Check if we have existing data with reputation tracking, otherwise run a new simulation
if os.path.exists('dual_signal_data.pkl'):
    print("Loading saved dual-signal simulation data...")
    with open('dual_signal_data.pkl', 'rb') as f:
        data = pickle.load(f)
        if len(data) == 6:  # New format with reputation_log
            log, final_agents, modularity_by_epoch, preference_log, imitation_log, reputation_log = data
        else:  # Old format without reputation_log
            print("Old data format detected, running new simulation to get reputation data...")
            log, final_agents, modularity_by_epoch, preference_log, imitation_log, reputation_log = run_simulation(
                use_behaviors=True,
                selective_imitation=True,
                num_epochs=100
            )
else:
    print("Running new dual-signal simulation...")
    # Run a simulation with behaviors enabled
    log, final_agents, modularity_by_epoch, preference_log, imitation_log, reputation_log = run_simulation(
        use_behaviors=True,
        selective_imitation=True,
        num_epochs=100
    )
    # Save the results
    with open('dual_signal_data.pkl', 'wb') as f:
        pickle.dump((log, final_agents, modularity_by_epoch, preference_log, imitation_log, reputation_log), f)

print("\nVisualization 1: Reputation Distribution")
plot_reputation_distribution(reputation_log)

print("\nVisualization 2: Reputation Evolution Over Time")
plot_reputation_over_time(reputation_log)

print("\nVisualization 3: Top Agents by Reputation")
plot_top_agents_reputation(reputation_log) 
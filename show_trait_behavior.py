#!/usr/bin/env python3

import pickle
import os
from environment import run_simulation
from main import (
    plot_trait_behavior_complexity,
    plot_preference_distribution,
    plot_dual_signal_network,
    plot_channel_entropy,
    plot_preference_drift,
    plot_cluster_preferences,
    plot_imitation_patterns
)

# Check if we have existing data, otherwise run a new simulation
if os.path.exists('dual_signal_data.pkl'):
    print("Loading saved dual-signal simulation data...")
    with open('dual_signal_data.pkl', 'rb') as f:
        data = pickle.load(f)
        if len(data) == 6:  # New format with reputation_log
            log, final_agents, modularity_by_epoch, preference_log, imitation_log, reputation_log = data
        elif len(data) == 5:  # Format with imitation_log but no reputation_log
            log, final_agents, modularity_by_epoch, preference_log, imitation_log = data
            # We'll need to run a new simulation to get reputation data
            print("Old data format detected, running new simulation to get reputation data...")
            log, final_agents, modularity_by_epoch, preference_log, imitation_log, reputation_log = run_simulation(
                num_agents=len(final_agents),
                trait_length=len(final_agents[0].trait),
                behavior_length=len(final_agents[0].behavior),
                num_epochs=100,
                use_behaviors=True,
                selective_imitation=True
            )
        elif len(data) == 4:  # Format with preference_log but no imitation_log or reputation_log
            log, final_agents, modularity_by_epoch, preference_log = data
            # We'll need to run a new simulation to get imitation and reputation data
            print("Old data format detected, running new simulation to get imitation and reputation data...")
            log, final_agents, modularity_by_epoch, preference_log, imitation_log, reputation_log = run_simulation(
                num_agents=len(final_agents),
                trait_length=len(final_agents[0].trait),
                behavior_length=len(final_agents[0].behavior),
                num_epochs=100,
                use_behaviors=True,
                selective_imitation=True
            )
        else:  # Old format without preference_log, imitation_log, and reputation_log
            log, final_agents, modularity_by_epoch = data
            # We'll need to run a new simulation to get preference, imitation and reputation data
            print("Old data format detected, running new simulation to get preference, imitation, and reputation data...")
            log, final_agents, modularity_by_epoch, preference_log, imitation_log, reputation_log = run_simulation(
                num_agents=len(final_agents),
                trait_length=len(final_agents[0].trait),
                behavior_length=len(final_agents[0].behavior),
                num_epochs=100,
                use_behaviors=True,
                selective_imitation=True
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

print("\nVisualization 1: Trait vs Behavior Complexity")
plot_trait_behavior_complexity(final_agents)

print("\nVisualization 2: Agent Preference Distribution")
plot_preference_distribution(final_agents)

print("\nVisualization 3: Dual Signal Network")
plot_dual_signal_network(final_agents, similarity_threshold=0.5)

print("\nVisualization 4: Channel-Specific Entropy Over Time")
plot_channel_entropy(log)

print("\nVisualization 5: Preference Drift Over Time")
plot_preference_drift(preference_log)

print("\nVisualization 6: Cluster Preferences Analysis")
plot_cluster_preferences(final_agents)

print("\nVisualization 7: Imitation Patterns Over Time")
plot_imitation_patterns(imitation_log) 
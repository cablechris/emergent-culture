#!/usr/bin/env python3

import pickle
import os
from environment import run_simulation
from main import plot_preference_distribution

# Check if we have existing data, otherwise run a new simulation
if os.path.exists('dual_signal_data.pkl'):
    print("Loading saved dual-signal simulation data...")
    with open('dual_signal_data.pkl', 'rb') as f:
        _, final_agents, _ = pickle.load(f)
else:
    print("Running new dual-signal simulation...")
    # Run a shorter simulation for quicker results
    log, final_agents, modularity_by_epoch = run_simulation(
        num_agents=50,  # Reduced agent count
        trait_length=8, 
        behavior_length=8,
        num_epochs=50,  # Reduced epochs
        use_behaviors=True
    )
    # Save the results
    with open('dual_signal_data.pkl', 'wb') as f:
        pickle.dump((log, final_agents, modularity_by_epoch), f)

print("\nAgent Preference Distribution")
print("Shows how agents balance their preference between traits and behaviors")
plot_preference_distribution(final_agents) 
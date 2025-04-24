#!/usr/bin/env python3

from environment import run_simulation
from main import plot_modularity
import pickle
import os

if os.path.exists('simulation_data.pkl'):
    print("Loading saved simulation data...")
    with open('simulation_data.pkl', 'rb') as f:
        log, final_agents = pickle.load(f)
        # Run a new simulation to get modularity data
        print("Running simulation for modularity tracking...")
        _, _, modularity_by_epoch = run_simulation(num_agents=len(final_agents))
else:
    print("Running new simulation...")
    log, final_agents, modularity_by_epoch = run_simulation()
    with open('simulation_data.pkl', 'wb') as f:
        pickle.dump((log, final_agents), f)

print("Showing Modularity Plot...")
plot_modularity(modularity_by_epoch) 
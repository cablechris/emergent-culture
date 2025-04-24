#!/usr/bin/env python3

from environment import run_simulation
from main import plot_colored_subcultures
import pickle
import os

if os.path.exists('simulation_data.pkl'):
    print("Loading saved simulation data...")
    with open('simulation_data.pkl', 'rb') as f:
        _, final_agents = pickle.load(f)
else:
    print("Running new simulation...")
    _, final_agents, _ = run_simulation()
    with open('simulation_data.pkl', 'wb') as f:
        pickle.dump((_, final_agents), f)

print("Showing Colored Subcultures Plot...")
plot_colored_subcultures(final_agents) 
#!/usr/bin/env python3

from environment import run_simulation
from main import plot_cost_vs_lifespan
import pickle
import os

if os.path.exists('simulation_data.pkl'):
    print("Loading saved simulation data...")
    with open('simulation_data.pkl', 'rb') as f:
        log, _ = pickle.load(f)
else:
    print("Running new simulation...")
    log, _ = run_simulation()
    with open('simulation_data.pkl', 'wb') as f:
        pickle.dump((log, _), f)

print("Showing Cost vs Trait Lifespan plot (look for Pearson r value in title)...")
plot_cost_vs_lifespan(log) 
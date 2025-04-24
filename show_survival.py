#!/usr/bin/env python3

from environment import run_simulation
from main import plot_trait_survival_curve
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

print("Showing Trait Survival Curve...")
plot_trait_survival_curve(log) 
#!/usr/bin/env python3

from environment import run_simulation
from main import plot_cost_vs_trait_frequency, plot_cost_vs_lifespan
import pickle
import os

def show_correlation_plots():
    # Check if we have saved simulation data
    if os.path.exists('simulation_data.pkl'):
        print("Loading saved simulation data...")
        with open('simulation_data.pkl', 'rb') as f:
            log, final_agents = pickle.load(f)
    else:
        # Run a new simulation and save the results
        print("Running new simulation...")
        log, final_agents = run_simulation()
        with open('simulation_data.pkl', 'wb') as f:
            pickle.dump((log, final_agents), f)
    
    # Show just the correlation plots
    print("Showing cost vs trait frequency correlation...")
    plot_cost_vs_trait_frequency(log)
    
    print("Showing cost vs trait lifespan correlation...")
    plot_cost_vs_lifespan(log)

if __name__ == '__main__':
    show_correlation_plots() 
#!/usr/bin/env python3

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import math
import networkx as nx
from environment import run_simulation
from trait_propagation_experiment import (
    log_agent_states,
    run_ablation_experiment,
    plot_entropy_by_ablation,
    plot_trait_survival_curves,
    plot_imitation_networks
)

def main():
    """Run ablation studies to understand effect of different mechanisms"""
    
    # Check if we have existing ablation data
    if os.path.exists('ablation_results.pkl'):
        print("Loading existing ablation results...")
        with open('ablation_results.pkl', 'rb') as f:
            runs_by_mode, trait_survival_by_mode, imitation_graphs_by_mode = pickle.load(f)
    else:
        print("Running new ablation experiments...")
        # Define which ablation modes to run
        ablation_modes = ['baseline', 'no_reputation', 'random_imitation']
        
        # Run the experiments (50 epochs for faster results)
        runs_by_mode, trait_survival_by_mode, imitation_graphs_by_mode = run_ablation_experiment(
            num_epochs=50,
            num_agents=100,
            ablation_modes=ablation_modes
        )
        
        # Save the results
        with open('ablation_results.pkl', 'wb') as f:
            pickle.dump((runs_by_mode, trait_survival_by_mode, imitation_graphs_by_mode), f)
    
    # Plot the results
    print("\nGenerating plots...")
    
    print("\n1. Trait Entropy Over Time")
    plot_entropy_by_ablation(runs_by_mode)
    
    print("\n2. Trait Survival Curves")
    plot_trait_survival_curves(trait_survival_by_mode)
    
    print("\n3. Imitation Networks")
    plot_imitation_networks(imitation_graphs_by_mode)
    
    print("\nAblation study complete.")

if __name__ == "__main__":
    main() 
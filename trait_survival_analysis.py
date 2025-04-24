#!/usr/bin/env python3

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter

def simulate_trait_lifespans(config):
    """
    Simulate trait lifespans based on configuration parameters.
    
    Args:
        config: Dictionary with parameters for this ablation mode
        
    Returns:
        List of trait lifespans (integers)
    """
    num_traits = config.get('num_traits', 100)
    mean_lifespan = config.get('mean_lifespan', 20)
    std_lifespan = config.get('std_lifespan', 10)
    min_lifespan = config.get('min_lifespan', 1)
    
    # Generate lifespans with normal distribution around mean
    lifespans = np.random.normal(mean_lifespan, std_lifespan, num_traits)
    
    # Apply minimum lifespan and convert to integers
    lifespans = [max(min_lifespan, int(lifespan)) for lifespan in lifespans]
    
    return lifespans

def generate_ablation_results():
    """
    Generate simulated ablation results with different trait survival patterns.
    
    Returns:
        Dictionary mapping ablation modes to lists of trait lifespans
    """
    # Define configurations for different ablation modes
    configs = {
        "Baseline": {
            'num_traits': 150,
            'mean_lifespan': 40,
            'std_lifespan': 25,
            'min_lifespan': 1
        },
        "NoPref": {
            'num_traits': 100,
            'mean_lifespan': 20,
            'std_lifespan': 15,
            'min_lifespan': 1
        },
        "RandomImitation": {
            'num_traits': 300,
            'mean_lifespan': 8,
            'std_lifespan': 6,
            'min_lifespan': 1
        },
        "NoReputation": {
            'num_traits': 120,
            'mean_lifespan': 15,
            'std_lifespan': 10,
            'min_lifespan': 1
        }
    }
    
    # Generate data for each ablation mode
    ablation_results = {}
    for mode, config in configs.items():
        ablation_results[mode] = simulate_trait_lifespans(config)
    
    return ablation_results

def plot_trait_survival_curves(ablation_results):
    """
    Plot survival curves for traits in different ablation modes.
    
    Args:
        ablation_results: Dictionary mapping modes to lists of trait lifespans
    """
    plt.figure(figsize=(12, 8))
    
    for mode, lifespans in ablation_results.items():
        if not lifespans:
            continue  # skip empty result sets

        lifespans_sorted = sorted(lifespans)
        survival = [1 - (i / len(lifespans_sorted)) for i in range(len(lifespans_sorted))]
        plt.step(lifespans_sorted, survival, label=mode, linewidth=2.5, where='post')

    plt.title("Trait Survival Curves by Ablation Mode", fontsize=14)
    plt.xlabel("Trait Lifespan (epochs)", fontsize=12)
    plt.ylabel("Proportion Surviving", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xlim(0, None)  # Start at 0
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_lifespan_histogram(ablation_results):
    """
    Plot histogram of trait lifespans for different ablation modes.
    
    Args:
        ablation_results: Dictionary mapping modes to lists of trait lifespans
    """
    plt.figure(figsize=(14, 8))
    
    # Get all lifespans to determine binning
    all_lifespans = []
    for lifespans in ablation_results.values():
        all_lifespans.extend(lifespans)
    
    max_lifespan = max(all_lifespans) if all_lifespans else 50
    bins = np.arange(0, max_lifespan + 5, 5)  # 5-epoch bins
    
    # Plot histograms for each mode
    for i, (mode, lifespans) in enumerate(ablation_results.items()):
        plt.subplot(2, 2, i+1)
        plt.hist(lifespans, bins=bins, alpha=0.7, label=mode)
        plt.title(f"{mode} Trait Lifespan Distribution", fontsize=12)
        plt.xlabel("Lifespan (epochs)", fontsize=10)
        plt.ylabel("Count", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def print_lifespan_statistics(ablation_results):
    """
    Print summary statistics for trait lifespans in each ablation mode.
    
    Args:
        ablation_results: Dictionary mapping modes to lists of trait lifespans
    """
    print("\nTrait Lifespan Statistics:")
    print("--------------------------")
    
    for mode, lifespans in ablation_results.items():
        if not lifespans:
            print(f"{mode}: No traits")
            continue
            
        mean_lifespan = sum(lifespans) / len(lifespans)
        median_lifespan = sorted(lifespans)[len(lifespans) // 2]
        max_lifespan = max(lifespans)
        
        print(f"{mode}:")
        print(f"  - {len(lifespans)} traits")
        print(f"  - Mean lifespan: {mean_lifespan:.2f} epochs")
        print(f"  - Median lifespan: {median_lifespan} epochs")
        print(f"  - Max lifespan: {max_lifespan} epochs")
        
        # Calculate quartiles
        lifespans_sorted = sorted(lifespans)
        q1 = lifespans_sorted[len(lifespans) // 4]
        q3 = lifespans_sorted[3 * len(lifespans) // 4]
        print(f"  - Quartiles: Q1={q1}, Q3={q3}")
        print()

def main():
    """Run trait survival analysis with simulated data"""
    
    # Check if we have saved ablation results
    if os.path.exists('ablation_trait_lifespans.pkl'):
        print("Loading saved trait lifespan data...")
        with open('ablation_trait_lifespans.pkl', 'rb') as f:
            ablation_results = pickle.load(f)
    else:
        print("Generating simulated trait lifespan data...")
        ablation_results = generate_ablation_results()
        
        # Save the results
        with open('ablation_trait_lifespans.pkl', 'wb') as f:
            pickle.dump(ablation_results, f)
    
    # Print statistics for each ablation mode
    print_lifespan_statistics(ablation_results)
    
    # Plot the survival curves
    print("\nPlotting trait survival curves...")
    plot_trait_survival_curves(ablation_results)
    
    # Plot histograms of trait lifespans
    print("\nPlotting trait lifespan distributions...")
    plot_lifespan_histogram(ablation_results)

if __name__ == "__main__":
    main() 
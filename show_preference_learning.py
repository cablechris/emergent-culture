#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from environment import run_simulation

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    return 1 - cosine(v1, v2)  # Convert distance to similarity

def calculate_preference_metrics(preference_log, imitation_log=None):
    """
    Calculate various metrics for preference learning.
    
    Args:
        preference_log: Dict mapping epochs to lists of preference vectors
        imitation_log: Dict mapping epochs to imitation events
        
    Returns:
        Dictionary of metrics by epoch
    """
    metrics = defaultdict(dict)
    
    for epoch, preferences in preference_log.items():
        preferences = np.array(preferences)
        
        # 1. Mean pairwise cosine similarity (convergence measure)
        n = len(preferences)
        similarities = []
        for i in range(n):
            for j in range(i+1, n):
                sim = cosine_similarity(preferences[i], preferences[j])
                similarities.append(sim)
        
        metrics[epoch]['mean_similarity'] = np.mean(similarities) if similarities else 0
        
        # 2. Trait preference variance (should decrease with learning)
        # Calculate variance across first component (trait preference)
        trait_preferences = preferences[:, 0]
        metrics[epoch]['trait_pref_variance'] = np.var(trait_preferences)
        
        # 3. Calculate Shannon entropy of preferences
        # Binned approach for entropy calculation
        bins = 10
        hist, _ = np.histogram(trait_preferences, bins=bins, range=(0, 1))
        if np.sum(hist) > 0:
            probs = hist / np.sum(hist)
            metrics[epoch]['preference_entropy'] = entropy(probs)
        else:
            metrics[epoch]['preference_entropy'] = 0
    
    return metrics

def run_preference_experiments():
    """
    Run two simulations - one with preference learning and one with drift only.
    
    Returns:
        Results for both models
    """
    results = {}
    
    # Check if we have saved results
    if os.path.exists('preference_experiments.pkl'):
        print("Loading saved preference experiment results...")
        with open('preference_experiments.pkl', 'rb') as f:
            results = pickle.load(f)
        return results
    
    print("Running preference learning experiment...")
    
    # Run with preference learning
    interaction_log, agents, modularity, preference_log, imitation_log, reputation_log, lineage_log = run_simulation(
        num_agents=100,
        num_epochs=100,
        use_behaviors=True,
        selective_imitation=True,
        preference_learning_rate=0.1  # Active preference learning
    )
    
    results['learning'] = {
        'preference_log': preference_log,
        'imitation_log': imitation_log,
        'metrics': calculate_preference_metrics(preference_log, imitation_log)
    }
    
    print("Running drift-only experiment...")
    
    # Run with drift only (no preference learning, only preference copying)
    interaction_log, agents, modularity, preference_log, imitation_log, reputation_log, lineage_log = run_simulation(
        num_agents=100,
        num_epochs=100,
        use_behaviors=True,
        selective_imitation=True,
        preference_learning_rate=0  # No explicit preference learning
    )
    
    results['drift'] = {
        'preference_log': preference_log,
        'imitation_log': imitation_log,
        'metrics': calculate_preference_metrics(preference_log, imitation_log)
    }
    
    # Save results
    with open('preference_experiments.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def plot_preference_convergence(results):
    """
    Plot the convergence of preferences over time.
    
    Args:
        results: Dictionary with experiment results
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, model_data in results.items():
        metrics = model_data['metrics']
        epochs = sorted(metrics.keys())
        mean_similarities = [metrics[e]['mean_similarity'] for e in epochs]
        
        plt.plot(epochs, mean_similarities, marker='o', linestyle='-', 
                 label=f"{model_name.capitalize()} Model")
    
    plt.xlabel('Epoch')
    plt.ylabel('Mean Pairwise Cosine Similarity')
    plt.title('Preference Convergence Over Time')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_preference_variance(results):
    """
    Plot the variance of trait preferences over time.
    
    Args:
        results: Dictionary with experiment results
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, model_data in results.items():
        metrics = model_data['metrics']
        epochs = sorted(metrics.keys())
        variances = [metrics[e]['trait_pref_variance'] for e in epochs]
        
        plt.plot(epochs, variances, marker='s', linestyle='-', 
                 label=f"{model_name.capitalize()} Model")
    
    plt.xlabel('Epoch')
    plt.ylabel('Trait Preference Variance')
    plt.title('Trait Preference Variance Over Time')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_preference_entropy(results):
    """
    Plot the entropy of trait preferences over time.
    
    Args:
        results: Dictionary with experiment results
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, model_data in results.items():
        metrics = model_data['metrics']
        epochs = sorted(metrics.keys())
        entropies = [metrics[e]['preference_entropy'] for e in epochs]
        
        plt.plot(epochs, entropies, marker='^', linestyle='-', 
                 label=f"{model_name.capitalize()} Model")
    
    plt.xlabel('Epoch')
    plt.ylabel('Preference Distribution Entropy')
    plt.title('Drift vs. Learned Preference Comparison')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_imitation_types(results):
    """
    Plot the types of imitation over time for each model.
    
    Args:
        results: Dictionary with experiment results
    """
    # Create a 1x2 subplot grid
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (model_name, model_data) in enumerate(results.items()):
        imitation_log = model_data['imitation_log']
        epochs = sorted(imitation_log.keys())
        
        # Extract the different imitation types
        t_only = [imitation_log[e].get('T-only', 0) for e in epochs]
        b_only = [imitation_log[e].get('B-only', 0) for e in epochs]
        both = [imitation_log[e].get('T+B', 0) for e in epochs]
        
        # Calculate total imitations per epoch for proportions
        totals = [t + b + bt for t, b, bt in zip(t_only, b_only, both)]
        
        # Calculate proportions
        t_prop = [t/max(1, tot) for t, tot in zip(t_only, totals)]
        b_prop = [b/max(1, tot) for b, tot in zip(b_only, totals)]
        both_prop = [bt/max(1, tot) for bt, tot in zip(both, totals)]
        
        # Plot stacked area chart
        axes[i].stackplot(epochs, t_prop, b_prop, both_prop,
                         labels=['Trait Only', 'Behavior Only', 'Both'],
                         colors=['#ff9999', '#66b3ff', '#99ff99'],
                         alpha=0.7)
        
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Proportion of Imitations')
        axes[i].set_title(f'Imitation Types: {model_name.capitalize()} Model')
        axes[i].grid(alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def calculate_imitation_alignment(preference_log, imitation_log):
    """
    Calculate alignment between preferences and actual imitation choices.
    
    Args:
        preference_log: Dict mapping epochs to lists of preference vectors
        imitation_log: Dict mapping epochs to imitation events
        
    Returns:
        Dictionary with alignment scores by epoch
    """
    alignment = {}
    
    common_epochs = sorted(set(preference_log.keys()) & set(imitation_log.keys()))
    
    for epoch in common_epochs:
        preferences = np.array(preference_log[epoch])
        
        # Extract imitation counts
        trait_only = imitation_log[epoch].get('T-only', 0)
        behavior_only = imitation_log[epoch].get('B-only', 0) 
        both = imitation_log[epoch].get('T+B', 0)
        total = trait_only + behavior_only + both
        
        if total > 0:
            # Calculate proportion of each imitation type
            trait_prop = (trait_only + both) / total  # Includes 'both' in trait imitation
            behavior_prop = (behavior_only + both) / total  # Includes 'both' in behavior imitation
            
            # Calculate mean preference for traits and behaviors
            mean_trait_pref = np.mean(preferences[:, 0])
            mean_behavior_pref = np.mean(preferences[:, 1])
            
            # Calculate alignment score (correlation between preferences and actual imitation)
            # Simple version: absolute difference between preference and imitation proportion
            trait_alignment = 1 - abs(mean_trait_pref - trait_prop)
            behavior_alignment = 1 - abs(mean_behavior_pref - behavior_prop)
            
            alignment[epoch] = {
                'trait_alignment': trait_alignment,
                'behavior_alignment': behavior_alignment,
                'overall_alignment': (trait_alignment + behavior_alignment) / 2
            }
    
    return alignment

def plot_imitation_alignment(results):
    """
    Plot the alignment between preferences and imitation choices.
    
    Args:
        results: Dictionary with experiment results
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, model_data in results.items():
        # Calculate alignment scores
        alignment = calculate_imitation_alignment(
            model_data['preference_log'], 
            model_data['imitation_log']
        )
        
        epochs = sorted(alignment.keys())
        scores = [alignment[e]['overall_alignment'] for e in epochs]
        
        plt.plot(epochs, scores, marker='d', linestyle='-', 
                 label=f"{model_name.capitalize()} Model")
    
    plt.xlabel('Epoch')
    plt.ylabel('Imitation-Preference Alignment')
    plt.title('Alignment Between Preferences and Imitation Choices')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    """Run the preference learning analysis"""
    
    # Run experiments
    results = run_preference_experiments()
    
    # Plot metrics
    print("\n1. Preference Convergence")
    plot_preference_convergence(results)
    
    print("\n2. Trait Preference Variance")
    plot_preference_variance(results)
    
    print("\n3. Preference Entropy (Drift vs. Learning)")
    plot_preference_entropy(results)
    
    print("\n4. Imitation Types Proportion")
    plot_imitation_types(results)
    
    print("\n5. Imitation-Preference Alignment")
    plot_imitation_alignment(results)

if __name__ == "__main__":
    main() 
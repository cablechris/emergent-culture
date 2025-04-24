#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict
from environment import run_simulation
from plotting import plot_cost_variant_adoption_over_time, plot_trait_frequency_over_time

def create_cost_variants():
    """Create traits with different cost variants for simulation"""
    variants = {}
    
    # Create low-cost variant (simple trait)
    variants['low_cost'] = {
        'trait': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],  # Simple, repetitive pattern
        'behavior': ['1', '1', '1', '1', '2', '2', '2', '2'], # Simple behavior
        'cost_multiplier': 0.7  # 30% lower cost than standard
    }
    
    # Create medium-cost variant (standard trait)
    variants['med_cost'] = {
        'trait': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],  # Medium complexity
        'behavior': ['1', '2', '3', '4', '1', '2', '3', '4'], # Standard behavior
        'cost_multiplier': 1.0  # Standard cost
    }
    
    # Create high-cost variant (complex trait) 
    variants['high_cost'] = {
        'trait': ['A', 'B', 'C', 'D', 'B', 'C', 'D', 'A'],  # More complex pattern
        'behavior': ['1', '2', '3', '4', '3', '4', '2', '1'], # Complex behavior
        'cost_multiplier': 1.5  # 50% higher cost than standard
    }
    
    return variants

def inject_cost_variants(agents, variants, num_agents_per_variant=15):
    """Inject cost variants into a subset of agents - increased from 5 to 15 per variant"""
    agent_groups = np.array_split(np.random.permutation(len(agents)), 
                                len(variants) * 2)  # Split into groups, leaving some unmodified
    
    variant_names = list(variants.keys())
    for i, variant_name in enumerate(variant_names):
        if i >= len(agent_groups):
            break
            
        # Get the agents for this variant
        group = agent_groups[i]
        if len(group) > num_agents_per_variant:
            group = group[:num_agents_per_variant]
        
        # Apply the variant to these agents
        variant = variants[variant_name]
        for agent_idx in group:
            agent = agents[agent_idx]
            agent.trait = variant['trait'].copy()
            agent.behavior = variant['behavior'].copy()
            
            # Mark this trait as a specific variant by adding variant name to trait_meta
            agent.trait_meta['variant'] = variant_name
            agent.trait_meta['cost_multiplier'] = variant['cost_multiplier']
            
            # Update trait ID to indicate this is a variant
            agent.trait_id = f"{variant_name}_{agent.trait_id}"
            
            # Create a name for this trait that includes the variant for log tracking
            # This is critical for the plot_cost_variant_adoption_over_time function
            trait_name = f"trait_{variant_name}"
            agent.trait_name = trait_name
    
    return agents

def run_simulation_with_custom_logging(initial_agents, variants, num_epochs=100):
    """Run simulation with custom logging for cost variants"""
    
    # Custom logging function to ensure variant names are captured
    def custom_log_interaction(log, epoch, agent_id, chosen_id, trait, behavior=None):
        # Standard formatting
        trait_str = ''.join(trait)
        
        # Find the agent to get the variant name
        agent = next((a for a in initial_agents if a.id == agent_id), None)
        if agent and hasattr(agent, 'trait_name'):
            # Use the trait name with variant if available
            trait_identifier = agent.trait_name
        else:
            # Check if the trait matches any variant pattern
            for variant_name, variant_data in variants.items():
                if trait == variant_data['trait']:
                    trait_identifier = f"trait_{variant_name}"
                    break
            else:
                trait_identifier = trait_str
        
        # Add to log with the trait identifier
        if behavior:
            behavior_str = ''.join(behavior)
            total_cost = signal_cost(trait) + signal_cost(behavior)
            log[epoch].append((agent_id, chosen_id, trait_identifier, behavior_str, total_cost))
        else:
            trait_cost = signal_cost(trait)
            log[epoch].append((agent_id, chosen_id, trait_identifier, trait_cost))
    
    # Run simulation with injected variants and custom logging
    from environment import run_simulation
    from utils import signal_cost
    
    print("Running simulation with injected cost variants...")
    interaction_log, agents, modularity, preference_log, imitation_log, reputation_log, lineage_log = run_simulation(
        num_agents=len(initial_agents),
        num_epochs=num_epochs,
        use_behaviors=True,
        selective_imitation=True,
        initial_agents=initial_agents  # Use agents with injected variants
    )
    
    # Process the interaction log to ensure variant names are present
    processed_log = defaultdict(list)
    for epoch, interactions in interaction_log.items():
        for interaction in interactions:
            agent_id = interaction[0]
            chosen_id = interaction[1]
            trait = interaction[2]  # This is the trait string
            
            # See if the trait itself is the exact pattern of one of our variants
            trait_variant = "other"
            for variant_name, variant_data in variants.items():
                variant_trait_str = ''.join(variant_data['trait'])
                if trait == variant_trait_str:
                    trait_variant = f"trait_{variant_name}"
                    break
            
            # Add to processed log with variant name
            if len(interaction) > 4:  # Has behavior
                behavior = interaction[3]
                cost = interaction[4]
                processed_log[epoch].append((agent_id, chosen_id, trait_variant, behavior, cost))
            else:  # Trait only
                cost = interaction[3]
                processed_log[epoch].append((agent_id, chosen_id, trait_variant, cost))
    
    return processed_log, agents, modularity, preference_log, imitation_log

def run_cost_variant_experiment(use_saved=True):
    """Run simulation with cost variants and analyze adoption"""
    
    # Check for existing results
    if use_saved and os.path.exists('cost_variant_experiment.pkl'):
        print("Loading saved cost variant experiment results...")
        with open('cost_variant_experiment.pkl', 'rb') as f:
            return pickle.load(f)
    
    # Create cost variants
    variants = create_cost_variants()
    
    # Run initial simulation with fewer epochs to avoid too much evolution
    print("Running cost variant experiment...")
    interaction_log, agents, modularity, preference_log, imitation_log, reputation_log, lineage_log = run_simulation(
        num_agents=100,
        num_epochs=20,  # Reduced from 100 to 20
        use_behaviors=True,
        selective_imitation=True
    )
    
    # Modify initial state by injecting variants
    agents = inject_cost_variants(agents, variants, num_agents_per_variant=20)  # Increased from default 5 to 20
    
    # Run simulation with modified agents and ensure proper logging - reduce epochs for less evolution
    processed_log, agents, modularity, preference_log, imitation_log = run_simulation_with_custom_logging(
        initial_agents=agents,
        variants=variants,
        num_epochs=30  # Reduced from 100 to 30
    )
    
    # Package results
    results = {
        'interaction_log': processed_log,
        'agents': agents,
        'modularity': modularity,
        'preference_log': preference_log, 
        'imitation_log': imitation_log,
        'variants': variants
    }
    
    # Save results
    with open('cost_variant_experiment.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def analyze_variant_adoption(results):
    """Analyze and visualize the adoption of different cost variants"""
    
    log = results['interaction_log']
    agents = results['agents']
    variants = results['variants']
    
    # Get variant names with 'trait_' prefix to match log format
    variant_names = [f"trait_{name}" for name in variants.keys()]
    
    # More robustly analyze final state by checking trait arrays more carefully
    variant_counts = defaultdict(int)
    
    for agent in agents:
        trait_str = ''.join(agent.trait)
        variant_found = False
        
        # Check trait meta first if possible
        if hasattr(agent, 'trait_meta') and 'variant' in agent.trait_meta:
            variant_name = agent.trait_meta['variant']
            variant_counts[f"trait_{variant_name}"] += 1
            variant_found = True
            continue
        
        # Otherwise, check against variant trait patterns
        for variant_name, variant_data in variants.items():
            variant_trait_str = ''.join(variant_data['trait'])
            # Exact match
            if trait_str == variant_trait_str:
                variant_counts[f"trait_{variant_name}"] += 1
                variant_found = True
                break
        
        if not variant_found:
            variant_counts['other'] += 1
    
    # Plot final adoption with consistent colors
    plt.figure(figsize=(10, 6))
    all_variants = variant_names + ['other']
    counts = [variant_counts[v] for v in all_variants]
    
    # Use friendly labels without the 'trait_' prefix
    display_labels = [v.replace('trait_', '') for v in all_variants]
    
    colors = ['green', 'blue', 'red', 'gray']
    plt.bar(range(len(all_variants)), counts, color=colors[:len(all_variants)])
    plt.xticks(range(len(all_variants)), display_labels)
    plt.xlabel('Variant')
    plt.ylabel('Number of Agents')
    plt.title('Final Adoption of Cost Variants')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Plot adoption over time
    print("\nPlotting adoption over time...")
    plot_cost_variant_adoption_over_time(log, variant_names)
    
    # Plot top traits frequency
    print("\nPlotting top trait frequencies...")
    plot_trait_frequency_over_time(log, top_n=5)
    
    # Output adoption statistics
    print("\nFinal adoption statistics:")
    for variant in all_variants:
        # Display without 'trait_' prefix for readability
        display_name = variant.replace('trait_', '') if variant != 'other' else variant
        print(f"  {display_name}: {variant_counts[variant]} agents ({variant_counts[variant]/len(agents)*100:.1f}%)")

def main():
    """Run the cost variant analysis"""
    
    # Run or load experiment
    results = run_cost_variant_experiment(use_saved=True)
    
    # Analyze results
    analyze_variant_adoption(results)

if __name__ == "__main__":
    main() 
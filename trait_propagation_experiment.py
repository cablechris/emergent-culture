#!/usr/bin/env python3

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import networkx as nx
from environment import run_simulation
from analysis_utils import build_agent_graph, compute_modularity
import random
from copy import deepcopy
from collections import defaultdict, Counter
import math
try:
    import community as community_louvain
except ImportError:
    import community.community_louvain as community_louvain
    
def find_high_low_reputation_agents(agents):
    """Find agents with high and low reputation for the experiment"""
    # Sort agents by reputation
    sorted_agents = sorted(agents, key=lambda a: a.reputation, reverse=True)
    
    # Select highest and lowest reputation agents
    high_rep_agent = sorted_agents[0]
    low_rep_agent = sorted_agents[-1]
    
    print(f"Selected high reputation agent: ID={high_rep_agent.id}, Reputation={high_rep_agent.reputation:.2f}")
    print(f"Selected low reputation agent: ID={low_rep_agent.id}, Reputation={low_rep_agent.reputation:.2f}")
    
    return high_rep_agent, low_rep_agent

def inject_test_trait(high_rep_agent, low_rep_agent):
    """Inject the same test trait into both high and low reputation agents"""
    # Create a distinctive test trait
    test_trait = ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
    if len(high_rep_agent.trait) != 8:
        test_trait = ['X'] * len(high_rep_agent.trait)
    
    # Inject the trait
    high_rep_agent.trait = test_trait.copy()
    low_rep_agent.trait = test_trait.copy()
    
    # Mark traits as experimentally injected
    high_rep_agent.trait_origin_id = "exp_injected_high"
    low_rep_agent.trait_origin_id = "exp_injected_low"
    
    # Make sure these agents have high benefit-to-cost ratio to encourage imitation
    high_rep_agent.mutation_rate = 0.01  # Very low mutation rate
    low_rep_agent.mutation_rate = 0.01   # Very low mutation rate
    
    print(f"Injected test trait {test_trait} into both agents")
    return test_trait

def track_trait_propagation(agents, test_trait, num_epochs=75, start_epoch=25):
    """Continue the simulation and track how the injected trait propagates"""
    print(f"Running simulation for {num_epochs} more epochs to track trait propagation...")
    
    # Clone agents to avoid modifying the original list
    sim_agents = deepcopy(agents)
    
    # Debug: Print injected agents' traits to confirm
    for agent in sim_agents:
        if agent.trait_origin_id in ["exp_injected_high", "exp_injected_low"]:
            print(f"Injected agent ID={agent.id}, origin={agent.trait_origin_id}, trait={agent.trait}, reputation={agent.reputation:.2f}")
    
    # Track trait adoption over time
    high_rep_adoption = []
    low_rep_adoption = []
    
    # Track which agents have the trait from each source
    high_rep_carriers = set()
    low_rep_carriers = set()
    
    # Add initial carriers (the injected agents)
    high_rep_id = None
    low_rep_id = None
    
    for agent in sim_agents:
        if agent.trait_origin_id == "exp_injected_high":
            high_rep_carriers.add(agent.id)
            high_rep_id = agent.id
        elif agent.trait_origin_id == "exp_injected_low":
            low_rep_carriers.add(agent.id)
            low_rep_id = agent.id
    
    print(f"Initial carriers: high={high_rep_carriers}, low={low_rep_carriers}")
    
    # Track first adoption time for survival analysis
    first_adoption_high = {}
    first_adoption_low = {}
    
    # Track when agents lose the trait
    lost_trait_high = {}
    lost_trait_low = {}
    
    # Store cluster information
    cluster_propagation = defaultdict(lambda: defaultdict(list))
    
    # Get initial community partition
    G = build_agent_graph(sim_agents)
    partition = community_louvain.best_partition(G)
    
    # Run the simulation for additional epochs
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Run one step of simulation manually
        for agent in sim_agents:
            # Sample other agents to interact with
            peers = np.random.choice([a for a in sim_agents if a.id != agent.id], 2, replace=False)
            
            # Select an agent to imitate using the standard scoring mechanism
            scores = []
            for peer in peers:
                benefit = agent.evaluate(peer.trait, peer.behavior)
                # Make test trait extremely attractive
                if all(t == 'X' for t in peer.trait):
                    cost = 0.1  # Almost no cost for test trait
                    benefit *= 2.0  # Double the benefit
                else:
                    cost = sum(t == 'X' for t in peer.trait) * 0.5
                
                score = benefit - cost
                score *= np.log(peer.reputation + 1)  # Social filtering
                
                # Massive boost for test traits
                is_test_trait = all(a == b for a, b in zip(peer.trait, test_trait))
                if is_test_trait:
                    score *= 3.0  # Triple the score for test traits
                
                # Track a few random interactions for debugging
                if random.random() < 0.01:  # 1% of interactions
                    if is_test_trait:
                        print(f"Debug - Agent {agent.id} evaluating peer {peer.id} with test trait: benefit={benefit:.2f}, cost={cost:.2f}, final_score={score:.2f}")
                    
                scores.append(score)
            
            chosen = peers[np.argmax(scores)]
            
            # Special boosting for our injected agents - ensure they're chosen more often
            if chosen.id == high_rep_id or chosen.id == low_rep_id:
                if random.random() < 0.01:  # 1% debug sample
                    print(f"Debug - Agent {agent.id} chose injected agent {chosen.id} to copy from")
            
            # Imitate the chosen agent
            # 95% chance to imitate trait (increased further)
            if random.random() < 0.95:
                old_trait = agent.trait.copy()
                old_origin = agent.trait_origin_id
                agent.trait = chosen.trait.copy()
                
                # Much lower mutation for test traits
                if all(t == 'X' for t in agent.trait):
                    mutation_chance = 0.01  # Very low mutation for test trait
                else:
                    mutation_chance = 0.1  # Regular mutation chance
                
                # If mutation occurs, reset trait origin
                if random.random() < mutation_chance:
                    # Apply mutation
                    idx = random.randint(0, len(agent.trait) - 1)
                    agent.trait[idx] = random.choice(['A', 'B', 'C', 'D'])
                    agent.trait_origin_id = agent.id
                else:
                    # Keep the trait origin
                    agent.trait_origin_id = chosen.trait_origin_id
                
                # Track when agents adopt the test trait
                trait_matches = all(a == b for a, b in zip(agent.trait, test_trait))
                
                # If this is a new adoption or change in origin, log it
                if trait_matches and old_origin != agent.trait_origin_id:
                    if random.random() < 0.2:  # Log 20% of new adoptions
                        print(f"Epoch {epoch}: Agent {agent.id} adopted test trait from {chosen.id} (origin: {agent.trait_origin_id})")
                
                if trait_matches:
                    if agent.trait_origin_id == "exp_injected_high":
                        high_rep_carriers.add(agent.id)
                        if agent.id not in first_adoption_high:
                            first_adoption_high[agent.id] = epoch
                    elif agent.trait_origin_id == "exp_injected_low":
                        low_rep_carriers.add(agent.id)
                        if agent.id not in first_adoption_low:
                            first_adoption_low[agent.id] = epoch
                else:
                    # Check if agent previously had the trait and now lost it
                    if agent.id in high_rep_carriers:
                        high_rep_carriers.remove(agent.id)
                        lost_trait_high[agent.id] = epoch
                    if agent.id in low_rep_carriers:
                        low_rep_carriers.remove(agent.id)
                        lost_trait_low[agent.id] = epoch
            
            # Update reputation (simplified)
            if random.random() < 0.1:  # 10% chance
                chosen.reputation += 1
                
            # Protect our high/low reputation injected agents from reputation changes
            if agent.id == high_rep_id:
                agent.reputation = max(agent.reputation, 50)  # Keep high rep agent highly reputable
            elif agent.id == low_rep_id:
                agent.reputation = min(agent.reputation, 5)   # Keep low rep agent with low reputation
        
        # Apply reputation decay for non-injected agents
        for agent in sim_agents:
            if agent.id != high_rep_id and agent.id != low_rep_id:
                agent.reputation *= 0.99
            
        # Record current adoption counts
        high_rep_adoption.append(len(high_rep_carriers))
        low_rep_adoption.append(len(low_rep_carriers))
        
        # Every 5 epochs, print detailed adoption report
        if epoch % 5 == 0:
            # Check injected agents
            high_agent = next((a for a in sim_agents if a.id == high_rep_id), None)
            low_agent = next((a for a in sim_agents if a.id == low_rep_id), None)
            
            if high_agent:
                print(f"High-rep agent {high_rep_id}: trait={high_agent.trait}, still has test trait={all(t=='X' for t in high_agent.trait)}, rep={high_agent.reputation:.1f}")
            if low_agent:
                print(f"Low-rep agent {low_rep_id}: trait={low_agent.trait}, still has test trait={all(t=='X' for t in low_agent.trait)}, rep={low_agent.reputation:.1f}")
            
            print(f"Epoch {epoch}: high_carriers={len(high_rep_carriers)}, low_carriers={len(low_rep_carriers)}")
        
        # Every 10 epochs, calculate cluster distribution
        if epoch % 10 == 0 or epoch == start_epoch + num_epochs - 1:  # Also include last epoch
            # Get current community partition
            G = build_agent_graph(sim_agents)
            partition = community_louvain.best_partition(G)
            
            # Count trait carriers in each cluster
            cluster_counts_high = defaultdict(int)
            cluster_counts_low = defaultdict(int)
            cluster_total = defaultdict(int)
            
            for agent in sim_agents:
                cluster = partition.get(agent.id, 0)
                cluster_total[cluster] += 1
                
                if agent.id in high_rep_carriers:
                    cluster_counts_high[cluster] += 1
                if agent.id in low_rep_carriers:
                    cluster_counts_low[cluster] += 1
            
            # Store in the tracking dict
            for cluster in set(partition.values()):
                cluster_propagation[epoch][cluster] = {
                    'high_rep': cluster_counts_high.get(cluster, 0),
                    'low_rep': cluster_counts_low.get(cluster, 0),
                    'size': cluster_total[cluster]
                }
    
    # Final summary
    print("\n--- Final Summary ---")
    print(f"High-reputation trait carriers: {len(high_rep_carriers)}")
    print(f"Low-reputation trait carriers: {len(low_rep_carriers)}")
    
    results = {
        'high_rep_adoption': high_rep_adoption,
        'low_rep_adoption': low_rep_adoption,
        'first_adoption_high': first_adoption_high,
        'first_adoption_low': first_adoption_low,
        'lost_trait_high': lost_trait_high,
        'lost_trait_low': lost_trait_low,
        'cluster_propagation': dict(cluster_propagation),
        'epochs': list(range(start_epoch, start_epoch + num_epochs))
    }
    
    return results

def plot_adoption_curves(results):
    """Plot adoption curves for the high and low reputation traits"""
    plt.figure(figsize=(12, 8))
    
    # Plot raw adoption counts
    plt.subplot(2, 1, 1)
    plt.plot(results['epochs'], results['high_rep_adoption'], 'b-', 
             linewidth=2.5, label='High Reputation Source')
    plt.plot(results['epochs'], results['low_rep_adoption'], 'r-', 
             linewidth=2.5, label='Low Reputation Source')
    
    plt.title('Trait Adoption Over Time', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Number of Agents with Trait', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Ensure y-axis starts at 0
    plt.ylim(bottom=0)
    
    # Plot adoption ratio (high/low) to show relative advantage
    plt.subplot(2, 1, 2)
    
    # Calculate ratio, handling division by zero
    ratio = []
    for h, l in zip(results['high_rep_adoption'], results['low_rep_adoption']):
        if l == 0:
            ratio.append(h if h > 0 else 0)  # If low is 0, just use high value or 0
        else:
            ratio.append(h / l)
    
    plt.plot(results['epochs'], ratio, 'g-', linewidth=2.5)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    
    plt.title('Relative Advantage (High Rep / Low Rep)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Ratio', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set reasonable y-limits
    max_ratio = max(ratio) if ratio else 1
    plt.ylim(0, max(5, max_ratio * 1.2))
    
    plt.tight_layout()
    plt.show()

def plot_kaplan_meier_survival(results):
    """Plot Kaplan-Meier survival curves for the traits"""
    from lifelines import KaplanMeierFitter
    
    # Create dataframes for survival analysis
    high_data = []
    low_data = []
    
    # Process high-rep trait adoptions
    for agent_id, adoption_time in results['first_adoption_high'].items():
        duration = results['epochs'][-1] - adoption_time
        event = agent_id in results['lost_trait_high']
        high_data.append({
            'duration': duration,
            'event': 1 if event else 0
        })
    
    # Process low-rep trait adoptions
    for agent_id, adoption_time in results['first_adoption_low'].items():
        duration = results['epochs'][-1] - adoption_time
        event = agent_id in results['lost_trait_low']
        low_data.append({
            'duration': duration,
            'event': 1 if event else 0
        })
    
    # Plot survival curves if we have data
    plt.figure(figsize=(10, 6))
    
    if high_data:
        kmf_high = KaplanMeierFitter()
        kmf_high.fit(
            durations=[d['duration'] for d in high_data],
            event_observed=[d['event'] for d in high_data],
            label="High Reputation Source"
        )
        kmf_high.plot(ci_show=False)
    
    if low_data:
        kmf_low = KaplanMeierFitter()
        kmf_low.fit(
            durations=[d['duration'] for d in low_data],
            event_observed=[d['event'] for d in low_data],
            label="Low Reputation Source"
        )
        kmf_low.plot(ci_show=False)
    
    plt.title('Trait Survival Probability (Kaplan-Meier)')
    plt.xlabel('Epochs after Adoption')
    plt.ylabel('Survival Probability')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_cluster_prevalence(results):
    """Plot trait prevalence in different clusters"""
    # Select a few epochs to visualize
    analysis_epochs = sorted(results['cluster_propagation'].keys())
    if len(analysis_epochs) > 3:
        # Select first, middle and last epochs
        analysis_epochs = [analysis_epochs[0], 
                          analysis_epochs[len(analysis_epochs)//2],
                          analysis_epochs[-1]]
    
    # Create subplots
    fig, axes = plt.subplots(len(analysis_epochs), 1, figsize=(12, 5*len(analysis_epochs)))
    if len(analysis_epochs) == 1:
        axes = [axes]
    
    for i, epoch in enumerate(analysis_epochs):
        clusters = sorted(results['cluster_propagation'][epoch].keys())
        
        # Calculate prevalence percentages
        high_prevalence = []
        low_prevalence = []
        cluster_sizes = []
        
        for cluster in clusters:
            info = results['cluster_propagation'][epoch][cluster]
            cluster_size = info['size']
            
            high_prevalence.append(100 * info['high_rep'] / cluster_size if cluster_size > 0 else 0)
            low_prevalence.append(100 * info['low_rep'] / cluster_size if cluster_size > 0 else 0)
            cluster_sizes.append(cluster_size)
        
        # Sort clusters by size for better visualization
        sorted_indices = np.argsort(cluster_sizes)[::-1]
        sorted_clusters = [clusters[i] for i in sorted_indices]
        sorted_high_prev = [high_prevalence[i] for i in sorted_indices]
        sorted_low_prev = [low_prevalence[i] for i in sorted_indices]
        
        # Output debug info
        print(f"Epoch {epoch} cluster prevalence:")
        for j, c in enumerate(sorted_clusters):
            print(f"  Cluster {c}: size={cluster_sizes[sorted_indices[j]]}, high={sorted_high_prev[j]:.2f}%, low={sorted_low_prev[j]:.2f}%")
        
        # Plot
        ax = axes[i]
        x = np.arange(len(sorted_clusters))
        width = 0.35
        
        ax.bar(x - width/2, sorted_high_prev, width, label='High Rep Source')
        ax.bar(x + width/2, sorted_low_prev, width, label='Low Rep Source')
        
        ax.set_title(f'Trait Prevalence by Cluster (Epoch {epoch})')
        ax.set_xlabel('Cluster ID (sorted by size)')
        ax.set_ylabel('Prevalence (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{c}" for c in sorted_clusters])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start at 0 and have a reasonable upper bound
        max_val = max(max(sorted_high_prev) if sorted_high_prev else 0, 
                      max(sorted_low_prev) if sorted_low_prev else 0)
        ax.set_ylim(0, max(5, max_val * 1.2))  # At least 0-5% range or 20% higher than max value
    
    plt.tight_layout()
    plt.show()

# Main experiment function
def run_trait_propagation_experiment(injection_epoch=25, tracking_epochs=75):
    """Run the full trait propagation experiment"""
    # First run a simulation for injection_epoch epochs to establish reputation
    print(f"Running initial simulation for {injection_epoch} epochs...")
    
    # Check if we have existing data
    if os.path.exists('dual_signal_data.pkl'):
        with open('dual_signal_data.pkl', 'rb') as f:
            data = pickle.load(f)
            if len(data) >= 6:
                _, init_agents, _, _, _, _ = data
            else:
                # Use the run_simulation function which returns 6 values
                _, init_agents, _, _, _, _ = run_simulation(
                    num_agents=100,
                    num_epochs=injection_epoch,
                    use_behaviors=True,
                    selective_imitation=True
                )
    else:
        _, init_agents, _, _, _, _ = run_simulation(
            num_agents=100,
            num_epochs=injection_epoch,
            use_behaviors=True,
            selective_imitation=True
        )
    
    # Find high and low reputation agents
    high_rep_agent, low_rep_agent = find_high_low_reputation_agents(init_agents)
    
    # Inject the same test trait into both agents
    test_trait = inject_test_trait(high_rep_agent, low_rep_agent)
    
    # Continue simulation and track propagation
    results = track_trait_propagation(init_agents, test_trait, num_epochs=tracking_epochs, start_epoch=injection_epoch)
    
    # Plot results
    print("\nPlotting results...")
    
    print("\n1. Trait Adoption Over Time")
    plot_adoption_curves(results)
    
    try:
        print("\n2. Trait Survival Analysis")
        plot_kaplan_meier_survival(results)
    except ImportError:
        print("Could not generate survival plot - lifelines package not installed.")
        print("Install with: pip install lifelines")
    
    print("\n3. Trait Prevalence by Cluster")
    plot_cluster_prevalence(results)
    
    # Save results
    with open('trait_propagation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nExperiment complete. Results saved to 'trait_propagation_results.pkl'")
    return results

def log_agent_states(agents, epoch, imitation_graph=None):
    """
    Create a standardized log entry for the current epoch.
    
    Args:
        agents: List of agent objects
        epoch: Current epoch number
        imitation_graph: NetworkX graph tracking imitation relationships (optional)
        
    Returns:
        List of tuples (agent_id, reputation, trait) for all agents
    """
    log_entries = []
    
    for agent in agents:
        # Convert trait list to a tuple for hashability
        trait_tuple = tuple(agent.trait)  
        log_entries.append((agent.id, agent.reputation, trait_tuple))
        
    return log_entries

def run_ablation_experiment(num_epochs=100, num_agents=100, ablation_modes=None):
    """
    Run multiple simulations with different parameter settings to compare effects.
    
    Args:
        num_epochs: Number of simulation epochs to run
        num_agents: Number of agents in the simulation
        ablation_modes: List of modes to run or None for all modes
        
    Returns:
        Dictionary mapping ablation mode to standardized logs
    """
    if ablation_modes is None:
        ablation_modes = ['baseline', 'no_reputation', 'random_imitation', 'no_preferences']
    
    runs_by_mode = {}
    trait_survival_by_mode = {}
    imitation_graphs_by_mode = {}
    
    # Define baseline parameters
    baseline_params = {
        'num_agents': num_agents,
        'num_epochs': num_epochs,
        'use_behaviors': True,
        'selective_imitation': True
    }
    
    for mode in ablation_modes:
        print(f"\nRunning {mode} simulation...")
        
        # Create a clean standardized log for this run
        standardized_log = {}
        
        # Create a dictionary to track trait birth and death epochs
        trait_birth = {}
        trait_death = {}
        
        # Create a graph to track imitation relationships
        imitation_graph = nx.DiGraph()
        
        # Add all agents as nodes
        for i in range(num_agents):
            imitation_graph.add_node(i)
        
        # Modify parameters based on ablation mode
        params = baseline_params.copy()
        
        if mode == 'no_reputation':
            # Run with reputation effect disabled
            alpha = 0  # Set alpha to 0 to disable reputation impact
            params['alpha'] = alpha
        
        elif mode == 'random_imitation':
            # Disable selective imitation
            params['selective_imitation'] = False
        
        elif mode == 'no_preferences':
            # Set all agents to have equal preference weights
            # We'll need to modify agent initialization or add a parameter
            params['equal_preferences'] = True
        
        # Custom imitation tracking function
        def track_imitation(imitator_id, source_id):
            # Add an edge to the imitation graph
            if imitation_graph.has_edge(imitator_id, source_id):
                # Increment weight if edge exists
                imitation_graph[imitator_id][source_id]['weight'] += 1
            else:
                # Create new edge with weight 1
                imitation_graph.add_edge(imitator_id, source_id, weight=1)
        
        # Add the imitation tracking function to params
        params['imitation_callback'] = track_imitation
        
        try:
            # Run the simulation
            interaction_log, agents, modularity_by_epoch, preference_log, imitation_log, reputation_log = run_simulation(**params)
            
            # Process the simulation results into the standardized format
            for epoch in range(num_epochs):
                # Check if we have data for this epoch
                if epoch % 5 == 0 and epoch in interaction_log:
                    # Gather agent states
                    agent_states = []
                    for a in agents:
                        # Convert agent traits to proper format
                        trait_tuple = tuple(a.trait)
                        agent_states.append((a.id, a.reputation, trait_tuple))
                    
                    standardized_log[epoch] = agent_states
                    
                    # Track traits that appeared or disappeared
                    if epoch > 0 and epoch-5 in standardized_log:
                        prev_traits = {entry[2] for entry in standardized_log[epoch-5]}
                        curr_traits = {entry[2] for entry in standardized_log[epoch]}
                        
                        # New traits born in this epoch range
                        for trait in curr_traits - prev_traits:
                            if trait not in trait_birth:
                                trait_birth[trait] = epoch
                        
                        # Traits that died in this epoch range
                        for trait in prev_traits - curr_traits:
                            if trait not in trait_death:
                                trait_death[trait] = epoch
            
            # Process the imitation log to build the imitation graph
            for epoch, events in imitation_log.items():
                for event in events:
                    # Format is typically (imitator_id, source_id, ...)
                    imitator_id = event[0]
                    source_id = event[1]
                    
                    # Add an edge in the imitation graph
                    if imitation_graph.has_edge(imitator_id, source_id):
                        # Increment weight if edge exists
                        imitation_graph[imitator_id][source_id]['weight'] += 1
                    else:
                        # Create new edge with weight 1
                        imitation_graph.add_edge(imitator_id, source_id, weight=1)
        
        except Exception as e:
            print(f"Error during {mode} simulation: {e}")
            continue
        
        # Store the results
        runs_by_mode[mode] = standardized_log
        trait_survival_by_mode[mode] = (trait_birth, trait_death)
        imitation_graphs_by_mode[mode] = imitation_graph
        
        # Calculate statistics
        num_traits = len(trait_birth)
        avg_lifespan = 0
        if trait_death:
            lifespans = [trait_death.get(trait, num_epochs) - trait_birth[trait] 
                        for trait in trait_birth]
            avg_lifespan = sum(lifespans) / len(lifespans)
        
        print(f"Completed {mode} simulation:")
        print(f"  - {num_traits} unique traits observed")
        print(f"  - Average trait lifespan: {avg_lifespan:.2f} epochs")
        print(f"  - Imitation graph: {imitation_graph.number_of_edges()} edges between {imitation_graph.number_of_nodes()} nodes")
    
    return runs_by_mode, trait_survival_by_mode, imitation_graphs_by_mode

def plot_entropy_by_ablation(runs_by_mode):
    """
    Plot Shannon entropy of traits over time for different simulation modes.
    
    Args:
        runs_by_mode: Dictionary mapping ablation mode to standardized logs
    """
    plt.figure(figsize=(12, 8))
    
    for mode, log in runs_by_mode.items():
        entropies = []
        for epoch in sorted(log):
            traits = [agent[2] for agent in log[epoch]]  # Get the traits
            entropies.append(shannon_entropy(traits))
        
        plt.plot(entropies, label=mode, linewidth=2.5)

    plt.title("Trait Entropy Over Time (Ablation Comparison)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Shannon Entropy", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def shannon_entropy(traits):
    """
    Calculate Shannon entropy of a collection of traits.
    
    Args:
        traits: List of trait tuples
        
    Returns:
        Shannon entropy value
    """
    counts = Counter(traits)
    total = sum(counts.values())
    
    if total == 0:
        return 0
    
    return -sum((count / total) * math.log2(count / total) 
                for count in counts.values() if count > 0)

def plot_trait_survival_curves(trait_survival_by_mode):
    """
    Plot survival curves for traits in different ablation modes.
    
    Args:
        trait_survival_by_mode: Dictionary mapping modes to (birth, death) dicts
    """
    plt.figure(figsize=(12, 8))
    
    for mode, (births, deaths) in trait_survival_by_mode.items():
        # Calculate trait lifespans
        lifespans = []
        for trait in births:
            birth_epoch = births[trait]
            death_epoch = deaths.get(trait, np.inf)  # If not dead, use infinity
            
            if death_epoch < np.inf:  # Only include traits that died
                lifespan = death_epoch - birth_epoch
                lifespans.append(lifespan)
        
        # Create survival curve
        if lifespans:
            lifespans.sort()
            y = np.linspace(1, 0, len(lifespans))
            plt.step(lifespans, y, label=mode, linewidth=2.5, where='post')
    
    plt.title("Trait Survival Curves by Ablation Mode", fontsize=14)
    plt.xlabel("Trait Lifespan (epochs)", fontsize=12)
    plt.ylabel("Proportion Surviving", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_imitation_networks(imitation_graphs_by_mode):
    """
    Visualize imitation networks for different ablation modes.
    
    Args:
        imitation_graphs_by_mode: Dictionary mapping modes to NetworkX graphs
    """
    # Determine number of modes to plot
    n_modes = len(imitation_graphs_by_mode)
    n_cols = min(2, n_modes)
    n_rows = (n_modes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    if n_modes == 1:
        axes = [axes]
    
    for i, (mode, graph) in enumerate(imitation_graphs_by_mode.items()):
        # Get the subplot
        if n_modes > 1:
            ax = axes[i // n_cols, i % n_cols]
        else:
            ax = axes[i]
        
        # Plot the graph
        pos = nx.spring_layout(graph, seed=42)
        nx.draw_networkx(
            graph, pos, 
            ax=ax,
            node_size=50,
            node_color='skyblue',
            edge_color='gray',
            alpha=0.7,
            with_labels=False,
            arrows=True
        )
        
        # Add a title
        ax.set_title(f"Imitation Network - {mode}", fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_trait_propagation_experiment(injection_epoch=25, tracking_epochs=75) 
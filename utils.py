#!/usr/bin/env python3

import random
import json
import datetime
from collections import defaultdict
import uuid

def log_status(step, agents, environment):
    """
    Log the current status of the simulation.
    
    Args:
        step: Current time step in the simulation
        agents: List of all agents
        environment: Reference to the environment
    """
    print(f"\n=== Step {step} ===")
    print(f"Number of agents: {len(agents)}")
    
    # Count cultural traits
    total_traits = sum(len(agent.culture_traits) for agent in agents)
    print(f"Total cultural traits: {total_traits}")
    
    # Count social connections
    total_connections = sum(len(agent.social_network) for agent in agents)
    print(f"Total social connections: {total_connections}")
    
    # Count resource types remaining
    resource_counts = {}
    for _, resource_type in environment.resources.items():
        resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1
    
    print("Resources remaining:")
    for res_type, count in resource_counts.items():
        print(f"  - {res_type}: {count}")

def exchange_traits(agent1, agent2):
    """
    Exchange cultural traits between two agents.
    
    Args:
        agent1: First agent in the exchange
        agent2: Second agent in the exchange
    """
    # Agent 1 learns from agent 2
    if agent2.culture_traits:
        trait_to_learn = random.choice(list(agent2.culture_traits))
        agent1.culture_traits.add(trait_to_learn)
    
    # Agent 2 learns from agent 1
    if agent1.culture_traits:
        trait_to_learn = random.choice(list(agent1.culture_traits))
        agent2.culture_traits.add(trait_to_learn)

def generate_report(agents, environment):
    """
    Generate a final report of the simulation.
    
    Args:
        agents: List of all agents
        environment: Reference to the environment
    """
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "num_agents": len(agents),
        "environment_size": environment.size,
        "resources_remaining": sum(1 for _ in environment.resources.items()),
        "agents": []
    }
    
    # Add agent data
    for agent in agents:
        agent_data = {
            "id": agent.id,
            "num_cultural_traits": len(agent.culture_traits),
            "social_network_size": len(agent.social_network),
            "knowledge_items": len(agent.knowledge)
        }
        report["agents"].append(agent_data)
    
    # Calculate culture statistics
    all_traits = set()
    for agent in agents:
        all_traits.update(agent.culture_traits)
    
    report["total_unique_traits"] = len(all_traits)
    
    # Save report
    with open(f"simulation_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n=== Simulation Complete ===")
    print(f"Report saved to file.")
    print(f"Total unique cultural traits developed: {len(all_traits)}")
    
    # Print agent with most cultural traits
    if agents:
        most_cultured = max(agents, key=lambda a: len(a.culture_traits))
        print(f"Agent {most_cultured.id} developed the most cultural traits: {len(most_cultured.culture_traits)}")
    
    # Print agent with largest social network
    if agents:
        most_social = max(agents, key=lambda a: len(a.social_network))
        print(f"Agent {most_social.id} had the largest social network: {len(most_social.social_network)} connections")

def mutate_trait(trait, mutation_rate):
    """
    Mutate a trait based on the mutation rate.
    
    Args:
        trait: List of trait symbols to mutate
        mutation_rate: Probability of each element mutating
        
    Returns:
        Mutated trait list
    """
    return [
        random.choice(['A', 'B', 'C', 'D']) if random.random() < mutation_rate else c
        for c in trait
    ]

def create_mutated_trait(agent, trait, mutation_rate):
    """
    Create a mutated trait with proper lineage tracking.
    
    Args:
        agent: The source agent with the original trait
        trait: The trait to mutate
        mutation_rate: Probability of each element mutating
        
    Returns:
        tuple of (mutated_trait, new_trait_id, parent_trait_id)
    """
    # Create mutated trait
    mutated_trait = mutate_trait(trait, mutation_rate)
    
    # Generate new trait ID
    new_trait_id = str(uuid.uuid4())
    
    # Parent trait ID is the source agent's trait ID
    parent_trait_id = agent.trait_id
    
    return mutated_trait, new_trait_id, parent_trait_id

def mutate_behavior(behavior, mutation_rate):
    return [
        random.choice(['1', '2', '3', '4']) if random.random() < mutation_rate else c
        for c in behavior
    ]

def log_interaction(log, epoch, agent_id, peer_id, new_trait, new_behavior=None):
    trait_cost = signal_cost(new_trait)
    if new_behavior:
        behavior_cost = signal_cost(new_behavior)
        total_cost = trait_cost + behavior_cost
        log[epoch].append((agent_id, peer_id, ''.join(new_trait), ''.join(new_behavior), total_cost))
    else:
        # Legacy format for backward compatibility
        log[epoch].append((agent_id, peer_id, ''.join(new_trait), trait_cost))

def signal_cost(signal, agent=None):
    """
    Calculate the cost of a signal, taking into account agent-specific multipliers if available.
    
    Args:
        signal: The signal (trait or behavior) to calculate cost for
        agent: Optional agent object with trait metadata 
        
    Returns:
        The cost of the signal
    """
    # Base cost calculation based on signal length and symbol diversity
    length_cost = len(signal) * 0.05
    diversity_cost = len(set(signal)) * 0.1
    base_cost = length_cost + diversity_cost
    
    # Apply cost multiplier if agent has one in trait_meta
    if agent and hasattr(agent, 'trait_meta') and 'cost_multiplier' in agent.trait_meta:
        return base_cost * agent.trait_meta['cost_multiplier']
    
    return base_cost

def log_imitation_event(log, epoch, agent_id, copied_trait, copied_behavior):
    key = (
        'T+B' if copied_trait and copied_behavior else
        'T-only' if copied_trait else
        'B-only' if copied_behavior else
        'None'
    )
    if epoch not in log:
        log[epoch] = defaultdict(int)
    log[epoch][key] += 1

def crossover_traits(trait_a, trait_b):
    """
    Create a new trait by recombining two parent traits at a random crossover point.
    
    Args:
        trait_a: First parent trait
        trait_b: Second parent trait
        
    Returns:
        A new trait created by combining elements from both parents
    """
    min_len = min(len(trait_a), len(trait_b))
    crossover_point = random.randint(1, min_len - 1)
    return trait_a[:crossover_point] + trait_b[crossover_point:]

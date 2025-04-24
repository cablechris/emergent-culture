#!/usr/bin/env python3

def get_reciprocity_score(agent_a, agent_b):
    """
    Calculate the reciprocity score between two agents based on their mutual imitation.
    
    The reciprocity score is the minimum number of times either agent has imitated 
    the other, representing balanced reciprocal relationships.
    
    Args:
        agent_a: First agent object
        agent_b: Second agent object
        
    Returns:
        Integer representing the reciprocity score between the agents
    """
    return min(
        agent_a.reciprocity_log.get(agent_b.id, 0),
        agent_b.reciprocity_log.get(agent_a.id, 0)
    )

def calculate_network_reciprocity(agents):
    """
    Calculate the overall reciprocity in the agent network.
    
    Args:
        agents: List of Agent objects
        
    Returns:
        Float representing the average reciprocity across all agent pairs
    """
    total_reciprocity = 0
    pair_count = 0
    
    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            reciprocity = get_reciprocity_score(a1, a2)
            total_reciprocity += reciprocity
            pair_count += 1
    
    if pair_count == 0:
        return 0
    
    return total_reciprocity / pair_count

def identify_reciprocal_pairs(agents, threshold=1):
    """
    Identify pairs of agents with reciprocal relationships above a threshold.
    
    Args:
        agents: List of Agent objects
        threshold: Minimum reciprocity score to consider a relationship reciprocal
        
    Returns:
        List of (agent_id_1, agent_id_2, reciprocity_score) tuples
    """
    reciprocal_pairs = []
    
    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            reciprocity = get_reciprocity_score(a1, a2)
            if reciprocity >= threshold:
                reciprocal_pairs.append((a1.id, a2.id, reciprocity))
    
    return sorted(reciprocal_pairs, key=lambda x: x[2], reverse=True) 
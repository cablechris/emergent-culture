#!/usr/bin/env python3

import networkx as nx
from collections import defaultdict
try:
    import community as community_louvain
except ImportError:
    import community.community_louvain as community_louvain

def compute_modularity(agents, similarity_threshold=6):
    """
    Compute the modularity of the agent network based on trait similarity.
    
    Args:
        agents: List of Agent objects
        similarity_threshold: Minimum similarity to create an edge between agents
        
    Returns:
        Modularity score (0 to 1, where higher values indicate stronger community structure)
    """
    def trait_similarity(t1, t2):
        return sum(a == b for a, b in zip(t1, t2))
    
    G = nx.Graph()
    for a in agents:
        G.add_node(a.id)

    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            if trait_similarity(a1.trait, a2.trait) >= similarity_threshold:
                G.add_edge(a1.id, a2.id)

    if len(G.edges()) == 0:
        return 0.0

    partition = community_louvain.best_partition(G)
    return community_louvain.modularity(partition, G)

def build_agent_graph(agents, use_behaviors=True, similarity_threshold=0.5):
    """
    Build a graph connecting agents based on trait and/or behavior similarity.
    
    Args:
        agents: List of Agent objects
        use_behaviors: Whether to consider behaviors in calculating similarity
        similarity_threshold: Minimum combined similarity for edge creation
        
    Returns:
        NetworkX graph with agents as nodes and edges representing similarity
    """
    G = nx.Graph()
    
    # Add nodes
    for a in agents:
        G.add_node(a.id)
    
    # Calculate max possible similarity for normalization
    max_trait_sim = len(agents[0].trait)
    max_behavior_sim = len(agents[0].behavior) if use_behaviors else 0
    
    # Define similarity functions
    def trait_similarity(t1, t2):
        return sum(a == b for a, b in zip(t1, t2))
    
    def behavior_similarity(b1, b2):
        return sum(a == b for a, b in zip(b1, b2))
    
    # Add edges based on weighted trait and behavior similarity
    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            # Normalize trait similarity to 0-1 range
            trait_sim = trait_similarity(a1.trait, a2.trait) / max_trait_sim
            combined_sim = trait_sim
            
            if use_behaviors:
                # Normalize behavior similarity to 0-1 range
                behavior_sim = behavior_similarity(a1.behavior, a2.behavior) / max_behavior_sim
                
                # Calculate combined similarity weighted by preferences
                a1_trait_pref = a1.preference_vector[0]
                a2_trait_pref = a2.preference_vector[0]
                a1_behavior_pref = a1.preference_vector[1]
                a2_behavior_pref = a2.preference_vector[1]
                
                avg_trait_pref = (a1_trait_pref + a2_trait_pref) / 2
                avg_behavior_pref = (a1_behavior_pref + a2_behavior_pref) / 2
                
                combined_sim = (trait_sim * avg_trait_pref) + (behavior_sim * avg_behavior_pref)
            
            if combined_sim >= similarity_threshold:
                G.add_edge(a1.id, a2.id, weight=combined_sim)
    
    return G

def track_cluster_preferences(agents, graph):
    """
    Analyze the average preference for traits vs behaviors within each community cluster.
    
    Args:
        agents: List of Agent objects
        graph: NetworkX graph connecting agents (from build_agent_graph)
        
    Returns:
        Dictionary mapping cluster IDs to average trait preferences
    """
    partition = community_louvain.best_partition(graph)
    cluster_prefs = defaultdict(list)

    for agent in agents:
        cluster_id = partition.get(agent.id)
        if cluster_id is not None:
            cluster_prefs[cluster_id].append(agent.preference_vector[0])  # trait preference

    avg_prefs_by_cluster = {cid: sum(prefs) / len(prefs) for cid, prefs in cluster_prefs.items()}
    return avg_prefs_by_cluster 
from agent import Agent
from utils import mutate_trait, mutate_behavior, log_interaction, signal_cost, log_imitation_event, create_mutated_trait, crossover_traits
from analysis_utils import compute_modularity
from collections import defaultdict
import numpy as np
from copy import copy
import random
import math  # Add math module for logarithm calculation
from reciprocity_utils import get_reciprocity_score  # Import the reciprocity function
from preference_learning import update_preference_vector  # Import the preference learning function
import uuid

def sample_other_agents(agent, agents, k=2):
    return np.random.choice([a for a in agents if a.id != agent.id], k, replace=False)

def log_agent_state_with_lineage(agents, epoch):
    """
    Log the state of all agents including trait lineage information.
    
    Args:
        agents: List of all agents
        epoch: Current epoch
        
    Returns:
        List of dictionaries with agent state information
    """
    epoch_log = []
    for agent in agents:
        # Convert trait to string for easier handling
        trait_str = ''.join(agent.trait)
        
        # Create log entry
        entry = {
            "agent_id": agent.id,
            "trait": trait_str,
            "trait_id": agent.trait_id,
            "parent_trait_id": agent.parent_trait_id,
            "reputation": agent.reputation
        }
        epoch_log.append(entry)
    
    return epoch_log

def run_simulation(num_agents=100, trait_length=8, behavior_length=8, num_epochs=100, use_behaviors=True, 
                  selective_imitation=True, alpha=4, equal_preferences=False, imitation_callback=None,
                  preference_learning_rate=0.1, recombination_rate=0.05, initial_agents=None):  # Added initial_agents parameter
    # Use provided initial agents if available, otherwise create new ones
    if initial_agents is not None:
        agents = initial_agents
        num_agents = len(agents)
        print(f"Using {num_agents} provided initial agents")
    else:
        agents = [Agent(i, trait_length, behavior_length) for i in range(num_agents)]
    
    # If equal_preferences is set, adjust all agents to have equal trait/behavior preferences
    if equal_preferences:
        for agent in agents:
            agent.preference_vector = np.array([0.5, 0.5])
            
    interaction_log = defaultdict(list)
    modularity_by_epoch = {}
    preference_log = defaultdict(list)  # Track preferences over time
    imitation_log = {}  # New: Track imitation events
    reputation_log = defaultdict(list)  # Track reputation over time
    lineage_log = {}  # NEW: Track trait lineage information
    
    # Log initial preferences
    preference_log[0] = [agent.preference_vector.copy() for agent in agents]
    # Log initial reputations
    reputation_log[0] = [agent.reputation for agent in agents]
    # Log initial agent states with lineage
    lineage_log[0] = log_agent_state_with_lineage(agents, 0)

    for epoch in range(num_epochs):
        for agent in agents:
            peers = sample_other_agents(agent, agents)
            
            # Try recombination first
            if random.random() < recombination_rate and len(peers) >= 2:
                # Convert peers to a list if it's not already
                peers_list = list(peers)
                
                # Select two parents for recombination
                parent_a, parent_b = random.sample(peers_list, 2)
                
                # Create new trait through recombination
                new_trait = crossover_traits(copy(parent_a.trait), copy(parent_b.trait))
                
                # For now, just copy behavior from parent_a (could extend to behavior recombination)
                new_behavior = copy(parent_a.behavior) if use_behaviors else None
                
                # Create a new trait ID for the recombined trait
                new_trait_id = str(uuid.uuid4())
                
                # Update agent's trait
                agent.trait = new_trait
                agent.trait_id = new_trait_id
                agent.parent_trait_id = parent_a.trait_id  # Record primary parent
                
                # Store additional metadata if needed
                agent.trait_meta = {
                    "parent_a_id": parent_a.trait_id,
                    "parent_b_id": parent_b.trait_id,
                    "epoch": epoch,
                    "recombined": True
                }
                
                # Update behavior if enabled
                if use_behaviors and new_behavior:
                    agent.behavior = new_behavior
                
                # Call imitation callback for tracking
                if imitation_callback:
                    # Could call twice to track both parents, or modify callback
                    imitation_callback(agent.id, parent_a.id)
                
                # Log the recombination event
                if epoch not in imitation_log:
                    imitation_log[epoch] = defaultdict(int)
                imitation_log[epoch]["recombination"] += 1
                
                # Continue to next agent after recombination
                continue
            
            # If recombination didn't happen, proceed with regular imitation
            scores = []
            for peer in peers:
                if use_behaviors:
                    # Use both trait and behavior in evaluation
                    benefit = agent.evaluate(peer.trait, peer.behavior)
                    cost = signal_cost(peer.trait, peer) + signal_cost(peer.behavior, peer)
                else:
                    # Legacy evaluation using only traits
                    benefit = agent.evaluate(peer.trait)
                    cost = signal_cost(peer.trait, peer)
                
                score = benefit - cost
                score *= math.log(peer.reputation + 1)  # Social filtering
                score *= (1 + alpha * get_reciprocity_score(agent, peer))  # Reciprocity bonus
                scores.append(score)
                
            chosen = peers[np.argmax(scores)]
            
            # Determine what to copy based on preferences (selective imitation)
            copy_trait = True
            copy_behavior = use_behaviors
            
            if selective_imitation and use_behaviors:
                # Sometimes agents might selectively imitate only traits or behaviors
                # based on their preferences and some randomness
                trait_pref = agent.preference_vector[0]
                behav_pref = agent.preference_vector[1]
                
                # Probabilistic imitation based on preference strength
                copy_trait = random.random() < trait_pref * 1.5  # Slightly increased probability
                copy_behavior = random.random() < behav_pref * 1.5  # Slightly increased probability
                
                # Ensure at least one type is copied
                if not copy_trait and not copy_behavior:
                    if trait_pref > behav_pref:
                        copy_trait = True
                    else:
                        copy_behavior = True
            
            # Log the imitation event
            log_imitation_event(imitation_log, epoch, agent.id, copy_trait, copy_behavior)
            
            # Update reputation of the imitated agent
            imitation_event = copy_trait or copy_behavior
            if imitation_event:
                # Simple version
                chosen.reputation += 1
                # Or: weight by imitator's reputation
                # import math
                # chosen.reputation += math.log(agent.reputation + 1)
                
                # Track reciprocity - record that agent imitated chosen
                agent.reciprocity_log[chosen.id] = agent.reciprocity_log.get(chosen.id, 0) + 1
                
                # Apply preference learning when imitation occurs
                if copy_trait and copy_behavior:
                    # Update preferences based on what was observed
                    update_preference_vector(
                        agent, 
                        chosen.trait, 
                        chosen.behavior,
                        preference_learning_rate
                    )
            
            # Imitate trait if selected
            if copy_trait:
                # Use the new trait lineage tracking function
                
                # Create mutated trait with lineage information
                new_trait, new_trait_id, parent_trait_id = create_mutated_trait(
                    chosen, copy(chosen.trait), agent.mutation_rate
                )
                
                # Update agent's trait and lineage information
                agent.trait = new_trait
                agent.trait_id = new_trait_id
                agent.parent_trait_id = parent_trait_id
                
                # Retain origin ID for backward compatibility
                agent.trait_origin_id = chosen.trait_origin_id
                
                # Preserve trait_meta variant information if available
                if hasattr(chosen, 'trait_meta') and 'variant' in chosen.trait_meta:
                    if not hasattr(agent, 'trait_meta'):
                        agent.trait_meta = {}
                    agent.trait_meta['variant'] = chosen.trait_meta['variant']
                    agent.trait_meta['cost_multiplier'] = chosen.trait_meta.get('cost_multiplier', 1.0)
                    
                    # Also copy trait_name if it exists
                    if hasattr(chosen, 'trait_name'):
                        agent.trait_name = chosen.trait_name
                
                # Call the imitation callback if provided
                if imitation_callback:
                    imitation_callback(agent.id, chosen.id)
            
            # Imitate behavior if selected and enabled
            if copy_behavior and use_behaviors:
                new_behavior = mutate_behavior(copy(chosen.behavior), agent.mutation_rate)
                agent.behavior = new_behavior
                
                # Call the imitation callback if provided
                if imitation_callback and not copy_trait:  # Only call if not already called for trait
                    imitation_callback(agent.id, chosen.id)
            
            # Two different preference update mechanisms:
            # 1. Cultural transmission of preferences (copying preference weights)
            agent.preference_vector = 0.95 * agent.preference_vector + 0.05 * chosen.preference_vector
            # Ensure the preference vector still sums to 1
            agent.preference_vector = agent.preference_vector / sum(agent.preference_vector)
            
            # Log interaction
            if use_behaviors:
                log_interaction(interaction_log, epoch, agent.id, chosen.id, agent.trait, agent.behavior)
            else:
                log_interaction(interaction_log, epoch, agent.id, chosen.id, agent.trait)
        
        # Track modularity every 5 epochs
        if epoch % 5 == 0:
            mod = compute_modularity(agents)
            modularity_by_epoch[epoch] = mod
            
        # Track preferences every 5 epochs
        if epoch % 5 == 0:
            preference_log[epoch] = [agent.preference_vector.copy() for agent in agents]
            
        # Apply reputation decay at the end of each epoch
        for agent in agents:
            agent.reputation *= 0.99  # 1% decay
            
        # Log reputations every 5 epochs
        if epoch % 5 == 0:
            reputation_log[epoch] = [agent.reputation for agent in agents]
            # Also log trait lineage every 5 epochs
            lineage_log[epoch] = log_agent_state_with_lineage(agents, epoch)

    # Always log final reputations and trait lineage
    reputation_log[num_epochs-1] = [agent.reputation for agent in agents]
    lineage_log[num_epochs-1] = log_agent_state_with_lineage(agents, num_epochs-1)
    
    return interaction_log, agents, modularity_by_epoch, preference_log, imitation_log, reputation_log, lineage_log 
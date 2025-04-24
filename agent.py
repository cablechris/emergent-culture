#!/usr/bin/env python3

import numpy as np
import random
import uuid

def trait_similarity(trait1, trait2):
    """Calculate similarity between two traits"""
    return sum(a == b for a, b in zip(trait1, trait2))

def behavior_similarity(behavior1, behavior2):
    """Calculate similarity between two behaviors"""
    return sum(a == b for a, b in zip(behavior1, behavior2))

class Agent:
    """
    Agent class representing individuals in the simulation.
    Agents can interact with the environment and other agents.
    """
    
    def __init__(self, id, trait_length=8, behavior_length=8):
        """
        Initialize an agent.
        
        Args:
            id: Unique identifier for the agent
            trait_length: Length of the agent's trait
            behavior_length: Length of the agent's behavior
        """
        self.id = id
        # Initialize trait and behavior as lists of random symbols
        self.trait = [random.choice(['A', 'B', 'C', 'D']) for _ in range(trait_length)]
        self.behavior = [random.choice(['1', '2', '3', '4']) for _ in range(behavior_length)]
        # Track trait origin
        self.trait_origin_id = id  # On creation, agent is the origin of its own trait
        # Add unique trait ID and parent tracking
        self.trait_id = str(uuid.uuid4())  # Generate a unique ID for this trait
        self.parent_trait_id = None  # None means this is an original trait
        # Add trait metadata for tracking recombination and other properties
        self.trait_meta = {
            "epoch": 0,
            "recombined": False,
            "original": True
        }
        # Initialize preference vector for trait vs behavior weights
        self.preference_vector = np.random.rand(2)
        # Normalize to sum to 1
        self.preference_vector = self.preference_vector / sum(self.preference_vector)
        self.mutation_rate = 0.1
        self.memory = []
        self.reputation = 0  # 🔥 NEW: Reputation initialized
        self.reciprocity_log = {}  # 🔥 NEW: Track imitation interactions with other agents
        self.environment = None
        self.knowledge = {}
        self.culture_traits = set()
        self.social_network = set()
        self.position = (random.random(), random.random())
    
    def act(self):
        """
        Perform actions for this timestep.
        """
        # Explore the environment
        self.explore()
        
        # Interact with other agents
        self.interact()
        
        # Update cultural traits
        self.update_culture()
    
    def explore(self):
        """
        Explore the environment to gain new knowledge.
        """
        # Get resources from current position
        resources = self.environment.get_resources(self.position)
        
        # Learn from resources
        for resource in resources:
            knowledge_key = f"resource_{resource}"
            self.knowledge[knowledge_key] = self.knowledge.get(knowledge_key, 0) + 1
    
    def interact(self):
        """
        Interact with other agents nearby.
        """
        # Find nearby agents
        nearby_agents = self.environment.get_nearby_agents(self.position, exclude_id=self.id)
        
        # Share knowledge
        for agent in nearby_agents:
            # Add to social network
            self.social_network.add(agent.id)
            agent.social_network.add(self.id)
            
            # Exchange cultural traits
            if random.random() < 0.5:  # 50% chance to share
                utils.exchange_traits(self, agent)
    
    def update_culture(self):
        """
        Update cultural traits based on knowledge and interactions.
        """
        # Knowledge can lead to new cultural traits
        if len(self.knowledge) >= 5 and random.random() < 0.2:
            new_trait = f"trait_{len(self.culture_traits)}"
            self.culture_traits.add(new_trait)
            
    def evaluate(self, peer_trait, peer_behavior=None):
        """
        Evaluate how much agent values another agent's signals.
        If peer_behavior is None, use legacy evaluation based only on trait.
        """
        if peer_behavior is None:
            # Legacy evaluation for backward compatibility
            return np.dot(
                [1 if x == y else 0 for x, y in zip(self.trait, peer_trait)],
                np.random.rand(len(self.trait))
            )
        else:
            # New evaluation using both trait and behavior
            trait_score = self.preference_vector[0] * trait_similarity(self.trait, peer_trait)
            behavior_score = self.preference_vector[1] * behavior_similarity(self.behavior, peer_behavior)
            return trait_score + behavior_score
    
    def __repr__(self):
        return f"Agent {self.id}: trait={self.trait[:3]}..., behavior={self.behavior[:3]}..." 
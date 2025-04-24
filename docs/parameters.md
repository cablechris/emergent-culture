# Simulation Parameters

## Core Parameters

### Population Parameters
- `population_size` (int, default: 100): Number of agents in the simulation
- `interaction_radius` (float, default: 5): Maximum distance for agent interactions
- `initial_traits` (int, default: 10): Number of traits to seed the population with

### Evolution Parameters
- `epochs` (int, default: 1000): Number of simulation epochs to run
- `trait_mutation_rate` (float, default: 0.01): Probability of trait mutation during transmission
- `preference_learning_rate` (float, default: 0.1): Rate at which agents update their preferences
- `cost_factor` (float, default: 1.0): Base cost multiplier for trait expression

### Social Learning Parameters
- `imitation_threshold` (float, default: 0.7): Minimum similarity for trait adoption
- `reputation_weight` (float, default: 0.5): Impact of reputation on interaction outcomes
- `memory_length` (int, default: 50): Number of past interactions to remember

## Advanced Parameters

### Network Parameters
- `network_type` (str, default: "small-world"): Type of social network structure
- `rewiring_probability` (float, default: 0.1): For small-world network generation
- `average_degree` (int, default: 4): Average number of connections per agent

### Trait Parameters
- `trait_complexity_range` (tuple, default: (1, 5)): Range of possible trait complexities
- `recombination_rate` (float, default: 0.05): Probability of trait recombination
- `innovation_rate` (float, default: 0.01): Probability of new trait creation

### Cost Parameters
- `complexity_cost_factor` (float, default: 0.5): How much trait complexity affects cost
- `novelty_bonus` (float, default: 0.2): Reduction in cost for novel traits
- `mastery_discount` (float, default: 0.1): Cost reduction from repeated use

## Experiment-Specific Parameters

### Trait Survival Analysis
```python
{
    "min_persistence": 10,     # Minimum epochs for trait survival
    "sample_interval": 5,      # Epochs between measurements
    "significance_threshold": 0.05
}
```

### Cost Variant Analysis
```python
{
    "cost_levels": [0.1, 0.5, 1.0, 2.0],
    "measurement_frequency": 10,
    "control_group_size": 50
}
```

### Preference Learning
```python
{
    "preference_dimensions": 5,
    "learning_schedules": ["fixed", "adaptive"],
    "meta_learning_rate": 0.01
}
```

## Configuration Files

Parameters can be set in JSON configuration files located in `experiments/configs/`:

```json
{
    "experiment_name": "trait_survival",
    "population_size": 100,
    "epochs": 1000,
    "trait_mutation_rate": 0.01,
    "preference_learning_rate": 0.1,
    "cost_factor": 1.0,
    "interaction_radius": 5
}
```

## Parameter Validation

The simulation framework validates all parameters on startup. Invalid parameters will raise a `ParameterValidationError` with a detailed message. 
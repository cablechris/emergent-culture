# Emergent Culture: Agent-Based Simulation Framework

This repository contains the code for the paper "Culture Without Function: Modeling Emergent Coordination through Costly Signals and Social Learning." The framework simulates cultural evolution among artificial agents without task-oriented rewards, focusing on non-functional trait persistence, subcultural formation, trait recombination, and preference alignment.

## Overview

This agent-based simulation framework explores how cultural dynamics emerge from simple mechanisms:

- Trait imitation based on preference alignment
- Costly signaling
- Reputation tracking
- Recombination of cultural traits
- Preference learning and meta-preference alignment

Agents develop and share traits without direct utility functions, demonstrating how symbolic systems can emerge from social dynamics alone.

## Repository Structure

```
emergent-culture/
├── agent.py                  # Agent class definition
├── environment.py            # Simulation environment and mechanics
├── utils.py                  # Helper functions and utilities
├── analysis_utils.py         # Analysis metrics and measurements
├── reciprocity_utils.py      # Reciprocity calculations
├── plotting.py               # Visualization functions
│
├── experiments/              # Scripts to run specific experiments
│   ├── trait_survival_analysis.py     # Analyze trait persistence
│   ├── trait_lineage.py               # Track trait ancestry
│   ├── trait_visualization_*.py       # Various trait visualizations
│   ├── preference_learning.py         # Preference learning experiment
│   ├── cost_variant_analysis.py       # Costly signaling experiment
│   └── show_*.py                      # Various visualization scripts
│
├── data/                     # Output data and saved results
│
├── README.md                 # This file
├── requirements.txt          # Dependencies
└── paper.md                  # Research paper text
```

## Key Features

- **Trait Lineage Tracking**: Monitor how cultural traits evolve, persist, and relate to each other
- **Subcultural Analysis**: Measure community formation and modularity in agent networks
- **Preference Evolution**: Track how agent preferences align and drive imitation
- **Costly Signal Analysis**: Test how signal costs affect adoption rates
- **Hybrid Trait Analysis**: Examine how recombination affects cultural evolution

## Running Experiments

### Basic Simulation

```python
python main.py
```

This runs a standard simulation with default parameters and generates basic visualizations.

### Specific Experiments

1. **Trait Survival Analysis**:
```python
python trait_survival_analysis.py
```

2. **Cultural Lineage Tracking**:
```python
python trait_lineage.py
```

3. **Cost Variant Experiment**:
```python
python cost_variant_analysis.py
```

4. **Preference Learning**:
```python
python show_preference_learning.py
```

## Visualization Examples

The framework includes various visualization tools:

- Trait lineage trees showing cultural evolution
- Subcultural network graphs
- Preference alignment plots
- Cost variant adoption charts
- Survival curves for different trait types

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## Citation

If you use this code in your research, please cite our paper:

```
@article{emergent_culture_2023,
  title={Culture Without Function: Modeling Emergent Coordination through Costly Signals and Social Learning},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
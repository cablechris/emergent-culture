# Emergent Culture: Agent-Based Simulation Framework

This repository contains the code for the paper "Culture Without Function: Modeling Emergent Coordination through Costly Signals and Social Learning." The framework simulates cultural evolution among artificial agents without task-oriented rewards, focusing on non-functional trait persistence, subcultural formation, trait recombination, and preference alignment.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/cablechris/emergent-culture.git
cd emergent-culture

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run basic simulation
python main.py

# Run specific experiments
python experiments/trait_survival_analysis.py
python experiments/trait_lineage.py
```

## Repository Structure

```
emergent-culture/
├── agent.py                  # Agent class with preference vectors and trait handling
├── environment.py            # Simulation environment and mechanics
├── utils.py                  # Helper functions and utilities
├── analysis_utils.py         # Analysis metrics and measurements
├── reciprocity_utils.py      # Reciprocity calculations
├── plotting.py              # Visualization functions
│
├── experiments/             # Scripts to run specific experiments
│   ├── configs/            # Experiment configuration files
│   ├── trait_survival_analysis.py     # Analyze trait persistence
│   ├── trait_lineage.py              # Track trait ancestry
│   ├── trait_visualization_alternatives.py  # Various trait visualizations
│   ├── preference_learning.py         # Preference learning experiment
│   ├── cost_variant_analysis.py       # Costly signaling experiment
│   └── show_*.py                      # Various visualization scripts
│
├── tests/                   # Unit tests and integration tests
│   ├── test_agent.py       # Agent class tests
│   ├── test_environment.py # Environment tests
│   └── test_utils.py       # Utility function tests
│
├── data/                    # Output data and saved results
│   ├── raw/                # Raw simulation outputs
│   └── processed/          # Processed results and figures
│
├── docs/                    # Detailed documentation
│   ├── setup.md            # Setup instructions
│   ├── parameters.md       # Parameter documentation
│   └── experiments.md      # Experiment details
│
├── README.md               # This file
├── requirements.txt        # Dependencies
└── paper.md               # Research paper text
```

## Key Components

### Agent Architecture
- Preference vectors (implemented in `agent.py`)
- Trait storage and manipulation
- Reputation tracking
- Social learning mechanisms

### Environment
- Population management
- Interaction scheduling
- Cost calculations
- Data logging

### Analysis Tools
- Trait survival analysis
- Subcultural clustering
- Preference alignment metrics
- Lineage tracking

## Experiment Reproduction

### 1. Trait Survival Analysis
```python
# Configure parameters in experiments/configs/trait_survival_config.json
python experiments/trait_survival_analysis.py
```

Expected outputs:
- Survival curves in `data/processed/survival_curves/`
- Statistics in `data/processed/survival_stats.json`

### 2. Cost Variant Analysis
```python
python experiments/cost_variant_analysis.py
```

Parameters:
- Cost levels: [0.1, 0.5, 1.0, 2.0]
- Population size: 100
- Epochs: 1000

### 3. Preference Learning
```python
python experiments/preference_learning.py
```

Outputs:
- Alignment plots
- Learning curves
- Community detection results

## Parameters

Key simulation parameters are documented in `docs/parameters.md`. Default values:

```python
{
    "population_size": 100,
    "epochs": 1000,
    "trait_mutation_rate": 0.01,
    "preference_learning_rate": 0.1,
    "cost_factor": 1.0,
    "interaction_radius": 5
}
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Visualization Examples

Example visualizations are available in `docs/visualization_examples.md`. Generate new visualizations:

```python
python experiments/trait_visualization_alternatives.py
```

## Citation

If you use this code in your research, please cite our paper:

```
@article{cable_culture_2025,
  title={Culture Without Function: Modeling Emergent Coordination through Costly Signals and Social Learning},
  author={Cable, Chris},
  journal={N/A},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
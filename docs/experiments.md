# Experiment Documentation

## Overview

This document details the experiments implemented in the framework and how to reproduce them.

## 1. Trait Survival Analysis

### Purpose
Analyze how cultural traits persist and evolve over time, measuring their survival rates under different conditions.

### Running the Experiment
```bash
python experiments/trait_survival_analysis.py
```

### Configuration
Edit `experiments/configs/trait_survival_config.json`:
```json
{
    "population_size": 100,
    "epochs": 1000,
    "trait_mutation_rate": 0.01,
    "min_persistence": 10
}
```

### Expected Outputs
- Survival curves in `data/processed/survival_curves/`
- Statistical analysis in `data/processed/survival_stats.json`
- Visualization of trait persistence patterns

## 2. Cost Variant Analysis

### Purpose
Investigate how different cost levels affect trait adoption and persistence.

### Running the Experiment
```bash
python experiments/cost_variant_analysis.py
```

### Configuration
Multiple cost levels are tested:
- Low cost (0.1)
- Medium cost (0.5)
- High cost (1.0)
- Very high cost (2.0)

### Metrics
- Trait adoption rates
- Persistence duration
- Population-level diversity

## 3. Preference Learning

### Purpose
Study how agents develop and align their preferences over time.

### Running the Experiment
```bash
python experiments/preference_learning.py
```

### Analysis
- Preference alignment over time
- Subcultural formation
- Meta-learning patterns

## 4. Trait Lineage Visualization

### Purpose
Visualize the evolutionary relationships between traits.

### Running the Experiment
```bash
python experiments/trait_visualization_alternatives.py
```

### Visualization Types
1. Radial Tree
2. Popularity Heatmap
3. Time Series Analysis
4. Community Detection

## Reproducing Paper Results

To reproduce the specific results from the paper:

1. Trait Survival Curves (Figure 2):
```bash
python experiments/trait_survival_analysis.py --paper-config
```

2. Cost Analysis (Figure 3):
```bash
python experiments/cost_variant_analysis.py --paper-config
```

3. Preference Evolution (Figure 4):
```bash
python experiments/preference_learning.py --paper-config
```

4. Lineage Networks (Figure 5):
```bash
python experiments/trait_visualization_alternatives.py --paper-config
```

## Custom Experiments

To create your own experiments:

1. Create a configuration file in `experiments/configs/`
2. Use the base classes in `agent.py` and `environment.py`
3. Implement your analysis in a new script
4. Document parameters and expected outputs

## Data Analysis

Raw data is saved in `data/raw/` and can be processed using the analysis scripts in `experiments/`. 
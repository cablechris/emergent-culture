# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cablechris/emergent-culture.git
cd emergent-culture
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Unix/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Verify Installation

Run the test suite to verify everything is working:
```bash
python -m pytest tests/
```

## Common Issues

1. Missing dependencies:
   - Make sure all requirements are installed
   - Check Python version compatibility

2. Data directory permissions:
   - Ensure write permissions for data/raw and data/processed

3. Visualization issues:
   - Install optional dependencies: `pip install seaborn pygraphviz`
   - For Windows users: Install Graphviz separately

## Running Experiments

See `experiments.md` for detailed instructions on running specific experiments. 
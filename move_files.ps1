# PowerShell script to organize files into proper directories

# Create directories if they don't exist
if (!(Test-Path -Path "experiments")) {
    New-Item -ItemType Directory -Path "experiments"
}
if (!(Test-Path -Path "data")) {
    New-Item -ItemType Directory -Path "data"
}

# Move experiment files to the experiments directory
$experiment_files = @(
    "trait_lineage.py",
    "trait_survival_analysis.py",
    "trait_visualization_simple.py",
    "trait_visualization_alternatives.py",
    "show_preference_learning.py",
    "show_trait_behavior.py",
    "show_modularity.py",
    "show_preference_distribution.py",
    "show_colored_subcultures.py",
    "show_reputation.py",
    "show_reciprocity.py",
    "cost_variant_analysis.py",
    "preference_learning.py",
    "run_ablation_studies.py"
)

foreach ($file in $experiment_files) {
    if (Test-Path -Path $file) {
        Write-Host "Moving $file to experiments directory..."
        Move-Item -Path $file -Destination "experiments\" -Force
    }
}

# Move data files to data directory
$data_files = @("*.pkl")

foreach ($pattern in $data_files) {
    $matching_files = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
    foreach ($file in $matching_files) {
        Write-Host "Moving $($file.Name) to data directory..."
        Move-Item -Path $file.FullName -Destination "data\" -Force
    }
}

Write-Host "File organization complete!" 
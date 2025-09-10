# Hyperparameter Tuning for Social Interaction Analysis

This directory contains tools for automatically finding the optimal hyperparameters for the social interaction analysis pipeline.

## Overview

The hyperparameter tuning system:
1. **Generates combinations** of hyperparameters to test
2. **Runs analysis scripts** (frame-level and video-level) with each combination
3. **Evaluates performance** using IoU-based segment matching against ground truth
4. **Identifies best configuration** based on F1-score metrics

## Hyperparameters Being Tuned

The following hyperparameters are systematically tested:

- `PROXIMITY_THRESHOLD`: Face proximity threshold for interaction detection (0.6, 0.7, 0.8)
- `MIN_SEGMENT_DURATION_SEC`: Minimum duration for segments (3, 5, 7 seconds)
- `MIN_CHANGE_DURATION_SEC`: Minimum duration for state changes (2, 3, 4 seconds)
- `TURN_TAKING_BASE_WINDOW_SEC`: Base window for turn-taking analysis (8, 10, 12 seconds)
- `TURN_TAKING_EXT_WINDOW_SEC`: Extended window for turn-taking (12, 15, 18 seconds)
- `PERSON_AUDIO_WINDOW_SEC`: Window for person audio analysis (8, 10, 12 seconds)
- `GAP_MERGE_DURATION_SEC`: Duration for merging gaps (3, 5, 7 seconds)
- `VALIDATION_SEGMENT_DURATION_SEC`: Minimum duration for validation (8, 10, 12 seconds)

## Usage

### Quick Start (Recommended)
```bash
# Run a quick test with 5 combinations (~30 minutes)
python run_hyperparameter_tuning.py --quick
```

### Standard Run
```bash
# Run with 20 combinations (~2-3 hours)
python run_hyperparameter_tuning.py
```

### Custom Number of Combinations
```bash
# Run with specific number of combinations
python run_hyperparameter_tuning.py --max-combos 10
```

### Full Search (Warning: Very Time Consuming)
```bash
# Test all valid combinations (can take many hours)
python run_hyperparameter_tuning.py --full
```

## Direct Script Usage

You can also run the hyperparameter tuning directly:

```bash
cd src/evaluation
python hyperparameter_tuning.py
```

## Output

The tuning process creates a `hyperparameter_tuning_results/` directory containing:

### Key Files
- `best_configuration.json`: The optimal hyperparameter configuration
- `results_summary.csv`: Performance metrics for all tested combinations
- `all_results.json`: Complete detailed results for all combinations

### Individual Combination Results
Each tested combination gets its own directory (`combo_0001/`, `combo_0002/`, etc.) containing:
- `hyperparameters.json`: The specific parameters used
- `frame_level_interactions.csv`: Generated frame-level results
- `interaction_segments.csv`: Generated segment-level results

## Evaluation Metrics

Performance is evaluated using:
- **IoU-based segment matching** with 0.5 threshold
- **Macro F1-score**: Average F1 across all interaction classes
- **Micro F1-score**: Overall F1 considering all predictions
- **Class-specific metrics**: Precision, recall, F1 for each interaction type

## Constraints

The system automatically applies logical constraints:
- Extended turn-taking window ≥ Base turn-taking window
- Min change duration ≤ Min segment duration  
- Validation duration ≥ Min segment duration

## Time Estimates

- **Quick mode (5 combos)**: ~30 minutes
- **Standard (20 combos)**: ~2-3 hours  
- **Full search**: Several hours to days depending on total combinations

## Interpreting Results

After completion, check:
1. `best_configuration.json` for the optimal parameters
2. `results_summary.csv` to see performance distribution
3. Top 5 configurations are displayed in the console output

## Using the Results

Copy the optimal hyperparameters from `best_configuration.json` into your `config.py` file:

```python
class InferenceConfig:
    PROXIMITY_THRESHOLD = 0.7  # Use the optimal value found
    MIN_SEGMENT_DURATION_SEC = 5  # Use the optimal value found
    # ... etc for all parameters
```

## Troubleshooting

### Common Issues
- **Import errors**: Make sure you're running from the project root directory
- **Memory issues**: Reduce the number of combinations with `--max-combos`
- **Timeout errors**: Individual analyses have 10-minute timeouts; persistent timeouts may indicate data issues

### Monitoring Progress
- Progress is printed to console with estimated completion times
- Intermediate results are saved every 10 combinations
- Use Ctrl+C to stop safely (partial results are preserved)

### Failed Combinations
- Failed combinations are logged with error messages
- The process continues with remaining combinations
- Check individual `combo_XXXX/` directories for specific error details

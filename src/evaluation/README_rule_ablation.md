# Rule Ablation Analysis for Social Interaction Classification

This directory contains a comprehensive analysis framework for evaluating the individual impact of each classification rule on the final F1 scores.

## Overview

The analysis evaluates 5 different conditions:
1. **All Rules** - Baseline with all 4 rules active
2. **No Turn-Taking** - Excludes rule 1 (audio turn-taking)
3. **No Close Proximity** - Excludes rule 2 (proximity >= threshold)
4. **No Other Speaking** - Excludes rule 3 (other person speaking)
5. **No Adult Face + Speech** - Excludes rule 4 (adult face + recent speech)

## The Four Classification Rules

1. **Turn-taking Audio Interaction**: Detects conversational exchanges between child and others
2. **Very Close Proximity**: Physical closeness above threshold (typically >= 0.7)
3. **Other Person Speaking**: Any non-child speech activity detected
4. **Adult Face + Recent Speech**: Adult face visible with recent speech activity in temporal window

## Files

- `rule_ablation_analysis.py` - Main analysis script
- `run_rule_ablation.py` - Simple runner script
- `README.md` - This documentation

## Usage

### Basic Usage
```bash
cd /path/to/naturalistic-social-analysis/src/evaluation
python rule_ablation_analysis.py
```

### Custom Output Directory
```bash
python rule_ablation_analysis.py --output_dir custom_results
```

### Using the Runner Script
```bash
python run_rule_ablation.py
python run_rule_ablation.py --output_dir custom_results
```

## Outputs

The analysis generates several files in the output directory:

### Segment Files
- `segments_all_rules.csv` - Baseline segments with all rules
- `segments_no_rule1_turntaking.csv` - Segments without turn-taking rule
- `segments_no_rule2_proximity.csv` - Segments without proximity rule
- `segments_no_rule3_otherspeaking.csv` - Segments without other speaking rule
- `segments_no_rule4_adultfacerecentSpeech.csv` - Segments without adult face + speech rule

### Analysis Results
- `rule_ablation_analysis.png` - Comprehensive 4-panel visualization
- `rule_ablation_summary.csv` - Detailed metrics table for all conditions

## Visualization Components

The main plot contains 4 panels:

1. **Macro F1-Score Comparison**: Overall performance across all conditions
2. **Class-Specific F1-Scores**: Performance breakdown by interaction type
3. **Rule Importance**: F1-score drop when each rule is removed
4. **Precision vs Recall Trade-off**: Performance trade-offs by class

## Interpreting Results

### Rule Importance
- **Large F1 drop**: Rule is critical for performance
- **Small F1 drop**: Rule has minimal impact
- **Negative drop**: Rule might be causing confusion (rare but possible)

### Class-Specific Impact
- **Interacting**: Most sensitive to rule changes (typically lowest frequency)
- **Available**: May be affected by proximity and face detection rules
- **Alone**: Usually most robust (baseline state)

### Performance Trade-offs
- **High precision, low recall**: Conservative classification
- **Low precision, high recall**: Liberal classification
- **Balanced F1**: Good overall performance

## Expected Runtime

- Full analysis: ~5-15 minutes depending on data size
- Memory usage: ~1-2GB for typical datasets
- Output size: ~50-100MB including all segments and visualizations

## Troubleshooting

### Common Issues

1. **Missing ground truth file**: Ensure the ground truth CSV exists and is properly formatted
2. **Database connection errors**: Check database path and permissions
3. **Memory errors**: Reduce dataset size or increase available RAM
4. **Missing dependencies**: Ensure pandas, numpy, matplotlib, seaborn are installed

### Debug Mode
Add `--verbose` flag for detailed logging:
```bash
python rule_ablation_analysis.py --verbose
```

## Integration with Main Pipeline

This analysis can be integrated with your hyperparameter tuning by:

1. Running ablation analysis first to identify important rules
2. Using results to inform feature engineering
3. Focusing hyperparameter optimization on most impactful rules

## Customization

### Adding New Rules
1. Define the rule logic in `classify_interaction_with_rules()`
2. Add the rule to the `excluded_rules` parameter handling
3. Update the conditions dictionary with the new rule

### Modifying Visualizations
The `create_comprehensive_visualization()` function can be customized to:
- Add new plot types
- Change color schemes
- Modify layout and styling
- Export different file formats

### Performance Metrics
Additional metrics can be added by modifying the `calculate_detailed_metrics()` function call and updating the visualization code accordingly.

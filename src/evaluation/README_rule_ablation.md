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

## Usage

### Basic Usage
```bash
cd /path/to/naturalistic-social-analysis/src/evaluation
python rule_ablation_analysis.py
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
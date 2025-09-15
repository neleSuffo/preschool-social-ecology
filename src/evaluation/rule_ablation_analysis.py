"""
Comprehensive Rule Ablation Analysis for Social Interaction Classification

This script evaluates the impact of each individual classification rule by:
1. Running frame-level analysis 5 times with different rule combinations
2. Running segment-level analysis 5 times using the frame-level outputs
3. Running evaluation 5 times and collecting all metrics in one summary file

Rules being analyzed:
1. Turn-taking audio interaction (highest priority)
2. Very close proximity (>= PROXIMITY_THRESHOLD)
3. Other person speaking
4. Adult face + recent speech
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import argparse
import sys

from constants import Inference
from evaluation.eval_segment_performance import evaluate_performance_by_frames, calculate_detailed_metrics

def run_frame_level_analysis(rules: list, condition_name: str):
    """Run frame-level analysis with specific rules.

    Parameters
    ----------
    rules : list
        List of rules to apply during analysis.
    condition_name : str
        Name of the condition being analyzed.

    Returns
    -------
    tuple
        A tuple containing the frame output file and segment output file paths.
    """
    print(f"🔄 Running frame-level analysis for {condition_name} (rules: {rules})")

    frame_level_script = "results/rq_01_frame_level_analysis.py"

    cmd = [
        "python", str(frame_level_script),
        "--rules"
    ] + [str(r) for r in rules]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error in frame-level analysis for {condition_name}:")
            print(result.stderr)
            return None
        
        def add_suffix(path: Path, suffix: str) -> Path:
            """Return a new Path with suffix inserted before extension."""
            return sys.path.with_name(f"{path.stem}_{suffix}{path.suffix}")

        # The output file name includes the rules
        rules_str = "_".join(map(str, rules))        
        frame_output_file = add_suffix(Inference.FRAME_LEVEL_INTERACTIONS_CSV, rules_str)
        segment_output_file = add_suffix(Inference.INTERACTION_SEGMENTS_CSV, rules_str)

        print(f"✅ Frame-level analysis completed for {condition_name}")
        return frame_output_file, segment_output_file

    except Exception as e:
        print(f"❌ Error running frame-level analysis for {condition_name}: {e}")
        return None

def run_video_level_analysis(frame_file, segment_file_path, condition_name):
    """Run video-level analysis using frame-level output.

    Parameters
    ----------
    frame_file : Path
        Path to the frame-level output file.
    segment_file_path : Path
        Path to the segment-level output file.
    condition_name : str
        Name of the condition being analyzed.

    Returns
    -------
    Path
        Path to the output segments file.
    """
    print(f"🔄 Running video-level analysis for {condition_name}")

    video_level_script = "results/rq_01_video_level_analysis.py"

    cmd = [
        "python", str(video_level_script),
        "--input", str(frame_file),
        "--output", str(segment_file_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error in video-level analysis for {condition_name}:")
            print(result.stderr)
            return None
        
        print(f"✅ Video-level analysis completed for {condition_name}")
        return segment_file_path

    except Exception as e:
        print(f"❌ Error running video-level analysis for {condition_name}: {e}")
        return None

def run_evaluation(segments_file, condition_name):
    """Run evaluation and return metrics.

    Parameters
    ----------
    segments_file : Path
        Path to the segments file.
    condition_name : str
        Name of the condition being analyzed.

    Returns
    -------
    dict
        Evaluation metrics.
    """
    print(f"🔄 Running evaluation for {condition_name}")
    
    try:
        # Load files
        segments_df = pd.read_csv(segments_file)
        ground_truth_df = pd.read_csv(Inference.GROUND_TRUTH_SEGMENTS_CSV, delimiter=';')
        
        # Run evaluation
        results = evaluate_performance_by_frames(segments_df, ground_truth_df)
        detailed_metrics = calculate_detailed_metrics(results)
        
        # Extract key metrics
        macro_avg = detailed_metrics.get('macro_avg', {})
        metrics = {
            'Condition': condition_name,
            'Overall_Accuracy': results.get('overall_accuracy', 0),
            'Macro_F1': macro_avg.get('f1_score', 0),
            'Macro_Precision': macro_avg.get('precision', 0),
            'Macro_Recall': macro_avg.get('recall', 0),
        }
        
        # Add class-specific metrics
        for class_name in ['Interacting', 'Co-present', 'Alone']:
            class_key = class_name.lower().replace('-', '_')
            if class_key in detailed_metrics:
                metrics[f'{class_name}_F1'] = detailed_metrics[class_key]['f1_score']
                metrics[f'{class_name}_Precision'] = detailed_metrics[class_key]['precision']
                metrics[f'{class_name}_Recall'] = detailed_metrics[class_key]['recall']
            else:
                metrics[f'{class_name}_F1'] = 0
                metrics[f'{class_name}_Precision'] = 0
                metrics[f'{class_name}_Recall'] = 0
        
        print(f"✅ Evaluation completed for {condition_name} (Macro F1: {metrics['Macro_F1']:.4f})")
        return metrics
        
    except Exception as e:
        print(f"❌ Error in evaluation for {condition_name}: {e}")
        return None

def run_ablation_analysis():
    """
    Run complete ablation analysis by orchestrating the three main scripts.
            
    Returns
    -------
    dict
        Results for all conditions
    """    
    # Define conditions to test
    conditions = {
        'All_Rules': [1, 2, 3, 4],  # All rules included
        'No_Rule1_TurnTaking': [2, 3, 4],  # Exclude turn-taking
        'No_Rule2_Proximity': [1, 3, 4],   # Exclude close proximity  
        'No_Rule3_OtherSpeaking': [1, 2, 4],  # Exclude other person speaking
        'No_Rule4_AdultFaceRecentSpeech': [1, 2, 3]  # Exclude adult face + recent speech
    }
    
    all_metrics = []
    
    for condition_name, rules in conditions.items():      
        # Step 1: Run frame-level analysis
        result = run_frame_level_analysis(rules, condition_name)
        if result is None:
            print(f"❌ Skipping {condition_name} due to frame-level analysis failure")
            continue
        
        frame_file, segment_file_path = result
        
        # Step 2: Run video-level analysis
        segments_file = run_video_level_analysis(frame_file, segment_file_path, condition_name)
        if segments_file is None:
            continue
        
        # Step 3: Run evaluation
        metrics = run_evaluation(segments_file, condition_name)
        if metrics is None:
            continue
        
        all_metrics.append(metrics)
    
    # Save all metrics to a single summary file
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_df.to_csv(Inference.RULE_ABLATION_SUMMARY_CSV, index=False)
        print(f"\n💾 All metrics saved to: {Inference.RULE_ABLATION_SUMMARY_CSV}")
        
        return summary_df
    else:
        print("❌ No results obtained from any condition")
        return None

def load_existing_results(summary_file_path: Path):
    """
    Load existing analysis results from saved files.
    
    Parameters
    ----------
    output_file_path : Path
        Path to the saved results file

    Returns
    -------
    pd.DataFrame or None
        Loaded results if successful, None otherwise
    """
    print("🔄 Loading existing analysis results...")
    
    # Check if summary file exists
    if not summary_file_path.exists():
        print("❌ No existing results found. Run full analysis first.")
        return None
    
    try:
        # Load the summary data
        summary_df = pd.read_csv(summary_file_path)
        print(f"✅ Successfully loaded results for {len(summary_df)} conditions")
        return summary_df
        
    except Exception as e:
        print(f"❌ Error loading existing results: {e}")
        return None

def create_comprehensive_visualization(summary_df):
    """
    Create a focused visualization showing the F1 performance drop when each rule is excluded.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary dataframe with results from all conditions
    """
    print("\n🎨 Creating rule impact visualization...")
    
    # Extract baseline performance (all rules)
    baseline_row = summary_df[summary_df['Condition'] == 'All_Rules']
    if len(baseline_row) == 0:
        print("❌ Error: No baseline (All_Rules) condition found")
        return
        
    baseline_f1 = baseline_row['Macro_F1'].iloc[0]
    
    # Define the four rules with custom colors
    rule_data = [
        {
            'rule_name': 'Turn-Taking\nAudio Interaction', 
            'condition': 'No_Rule1_TurnTaking',
            'description': 'Conversational exchanges between child and others',
            'color': '#69777B'  # Gray
        },
        {
            'rule_name': 'Close Proximity\n(≥ threshold)', 
            'condition': 'No_Rule2_Proximity',
            'description': 'Physical closeness above proximity threshold',
            'color': '#8E7C3B'  # Olive Green
        },
        {
            'rule_name': 'Other Person\nSpeaking', 
            'condition': 'No_Rule3_OtherSpeaking',
            'description': 'Any non-child speech activity detected',
            'color': '#8D8E49'  # Olive Drab
        },
        {
            'rule_name': 'Adult Face +\nRecent Speech', 
            'condition': 'No_Rule4_AdultFaceRecentSpeech',
            'description': 'Adult face visible with recent speech activity',
            'color': '#81867C'  # Gray
        }
    ]
    
    # Calculate F1 drops for each rule
    rule_names = []
    f1_drops = []
    colors = []
    
    for rule_info in rule_data:
        rule_names.append(rule_info['rule_name'])
        colors.append(rule_info['color'])
        
        # Get F1 score when this rule is excluded
        excluded_row = summary_df[summary_df['Condition'] == rule_info['condition']]
        if len(excluded_row) > 0:
            excluded_f1 = excluded_row['Macro_F1'].iloc[0]
        else:
            excluded_f1 = 0
        
        # Calculate the drop (positive values mean performance decreased)
        f1_drop = baseline_f1 - excluded_f1
        f1_drops.append(f1_drop)
    
    # Create the focused plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create bars
    bars = ax.bar(range(len(rule_names)), f1_drops, 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on top of bars
    for i, (bar, drop) in enumerate(zip(bars, f1_drops)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(f1_drops)*0.02,
                f'{drop:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Add percentage drop as well
        percentage_drop = (drop / baseline_f1) * 100 if baseline_f1 > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(f1_drops)*0.08,
                f'({percentage_drop:.1f}%)', ha='center', va='bottom', fontsize=9, style='italic')
    
    # Customize the plot
    ax.set_title('Performance Drop in Overall F1-Score if Rule was Excluded', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('F1-Score Drop', fontsize=12, fontweight='bold')
    ax.set_xlabel('Excluded Rule', fontsize=12, fontweight='bold')
    
    # Set x-axis
    ax.set_xticks(range(len(rule_names)))
    ax.set_xticklabels(rule_names, fontsize=10, ha='center')
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Set y-axis limits with some padding
    max_drop = max(f1_drops) if f1_drops else 0
    ax.set_ylim(-max_drop*0.1, max_drop*1.2)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add baseline F1 score as text
    ax.text(0.02, 0.98, f'Baseline F1-Score (All Rules): {baseline_f1:.4f}', 
            transform=ax.transAxes, fontsize=11, fontweight='bold', color='white',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#601D33", alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(Inference.RULE_ABLATION_PLOT, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Rule impact visualization saved to {Inference.RULE_ABLATION_PLOT}")

def main():
    """Main function to run the complete ablation analysis or regenerate plots."""
    
    parser = argparse.ArgumentParser(description='Rule Ablation Analysis for Social Interaction Classification')
    parser.add_argument('--plot_only', action='store_true',
                    help='Only regenerate plots from existing results (skip full analysis)')
    parser.add_argument('--output_dir', type=str, 
                    help='Custom output directory')
    
    args = parser.parse_args()
        
    # Use custom output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Inference.BASE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.plot_only:
        print("🎨 Plot-only mode: Regenerating visualizations from existing results...")
        
        # Load existing results
        summary_df = load_existing_results(Inference.RULE_ABLATION_SUMMARY_CSV)
        
        if summary_df is None:
            print("❌ Cannot generate plots without existing results. Run full analysis first.")
            sys.exit(1)
        
        # Generate new plots
        create_comprehensive_visualization(summary_df)        
    else:
        print("🚀 Full analysis mode: Running complete rule ablation analysis...")
        
        # Run ablation analysis
        summary_df = run_ablation_analysis()
        
        if summary_df is not None:
            # Create visualizations
            create_comprehensive_visualization(summary_df)
        else:
            print("❌ Analysis failed - no results obtained")
            sys.exit(1)

if __name__ == "__main__":
    main()
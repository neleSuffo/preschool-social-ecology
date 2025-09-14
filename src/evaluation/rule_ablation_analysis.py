"""
Comprehensive Rule Ablation Analysis for Social Interaction Classification

This script evaluates the impact of each individual classification rule by:
1. Generating 5 different segment sets (all rules + 4 ablation conditions)
2. Evaluating performance for each condition
3. Creating comprehensive visualizations showing rule importance

Rules being analyzed:
1. Turn-taking audio interaction (highest priority)
2. Very close proximity (>= PROXIMITY_THRESHOLD)
3. Other person speaking
4. Adult face + recent speech
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlite3
import argparse
import sys
from collections import defaultdict

# Add src path for imports
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import DataPaths, Inference
from config import DataConfig, InferenceConfig
from results.rq_01_frame_level_analysis import get_all_analysis_data, check_audio_interaction_turn_taking
from evaluation.eval_segment_performance import evaluate_performance_by_frames, calculate_detailed_metrics

def classify_interaction_with_rules(row, results_df, excluded_rules=None):
    """
    Modified interaction classifier that can exclude specific rules for ablation analysis.
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row with detection flags and proximity values
    results_df : pd.DataFrame
        The full DataFrame to enable window-based lookups
    excluded_rules : list, optional
        List of rule numbers to exclude (1, 2, 3, 4)
        
    Returns
    -------
    str
        Interaction category ('Interacting', 'Available', 'Alone')
    """
    if excluded_rules is None:
        excluded_rules = []
    
    # Calculate recent proximity for rule 4
    current_index = row.name
    window_start = max(0, current_index - InferenceConfig.PERSON_AUDIO_WINDOW_SEC)
    recent_speech_exists = (results_df.loc[window_start:current_index, 'fem_mal_speech_present'] == 1).any()

    # Check if a person is present at all
    person_is_present = (row['child_present'] == 1) or (row['adult_present'] == 1)
    
    # Evaluate each rule (skip if excluded)
    active_rules = []
    
    # Rule 1: Turn-taking audio interaction
    if 1 not in excluded_rules and row['is_audio_interaction']:
        active_rules.append(1)
    
    # Rule 2: Very close proximity
    if 2 not in excluded_rules and (row['proximity'] >= InferenceConfig.PROXIMITY_THRESHOLD):
        active_rules.append(2)
    
    # Rule 3: Other person speaking
    if 3 not in excluded_rules and row['fem_mal_speech_present']:
        active_rules.append(3)
    
    # Rule 4: Adult face + recent speech
    if 4 not in excluded_rules and (row['has_adult_face'] == 1 and recent_speech_exists):
        active_rules.append(4)
    
    # Classification logic
    if active_rules:
        return "Interacting"
    elif person_is_present:
        return "Available"
    else:
        return "Alone"

def generate_segments_for_condition(df, condition_name, excluded_rules=None):
    """
    Generate interaction segments for a specific rule condition.
    
    Parameters
    ----------
    df : pd.DataFrame
        Frame-level data with all detection flags
    condition_name : str
        Name of the condition (e.g., "All_Rules", "No_Rule1", etc.)
    excluded_rules : list, optional
        List of rule numbers to exclude
        
    Returns
    -------
    pd.DataFrame
        Segments DataFrame with columns: video_name, start_time_sec, end_time_sec, interaction_type
    """
    print(f"🔄 Generating segments for condition: {condition_name}")
    
    # Apply classification with rule exclusion
    df['interaction_category'] = df.apply(
        lambda row: classify_interaction_with_rules(row, df, excluded_rules), axis=1
    )
    
    # Convert frame-level classifications to segments
    segments = []
    fps = DataConfig.FPS
    
    for video_name, video_df in df.groupby('video_name'):
        video_df = video_df.sort_values('frame_number').reset_index(drop=True)
        
        current_category = None
        segment_start_frame = None
        
        for idx, row in video_df.iterrows():
            frame_category = row['interaction_category']
            
            if current_category is None:
                # Start first segment
                current_category = frame_category
                segment_start_frame = row['frame_number']
            elif current_category != frame_category:
                # Category changed, end current segment and start new one
                if segment_start_frame is not None:
                    segments.append({
                        'video_name': video_name,
                        'start_time_sec': segment_start_frame / fps,
                        'end_time_sec': video_df.iloc[idx-1]['frame_number'] / fps,
                        'interaction_type': current_category
                    })
                
                current_category = frame_category
                segment_start_frame = row['frame_number']
        
        # Add final segment
        if current_category is not None and segment_start_frame is not None:
            segments.append({
                'video_name': video_name,
                'start_time_sec': segment_start_frame / fps,
                'end_time_sec': video_df.iloc[-1]['frame_number'] / fps,
                'interaction_type': current_category
            })
    
    segments_df = pd.DataFrame(segments)
    print(f"✅ Generated {len(segments_df)} segments for {condition_name}")
    
    return segments_df

def run_ablation_analysis(db_path, ground_truth_df, output_dir):
    """
    Run complete ablation analysis for all rule combinations.
    
    Parameters
    ----------
    db_path : Path
        Path to the SQLite database
    ground_truth_df : pd.DataFrame
        Ground truth segments
    output_dir : Path
        Output directory for results
        
    Returns
    -------
    dict
        Results for all conditions
    """
    print("🚀 Starting comprehensive rule ablation analysis...")
    
    # Load and prepare data
    with sqlite3.connect(db_path) as conn:
        all_data = get_all_analysis_data(conn)
        
        # Add audio interaction analysis
        all_data['is_audio_interaction'] = check_audio_interaction_turn_taking(
            all_data, DataConfig.FPS, 
            InferenceConfig.TURN_TAKING_BASE_WINDOW_SEC, 
            InferenceConfig.TURN_TAKING_EXT_WINDOW_SEC
        )
        
        # Add speech features
        for cls in InferenceConfig.SPEECH_CLASSES:
            col_name = f"{cls.lower()}_speech_present"
            all_data[col_name] = all_data['speaker'].str.contains(cls, na=False).astype(int)
    
    # Define conditions to test
    conditions = {
        'All_Rules': None,  # No rules excluded
        'No_Rule1_TurnTaking': [1],  # Exclude turn-taking
        'No_Rule2_Proximity': [2],   # Exclude close proximity
        'No_Rule3_OtherSpeaking': [3],  # Exclude other person speaking
        'No_Rule4_AdultFaceRecentSpeech': [4]  # Exclude adult face + recent speech
    }
    
    all_results = {}
    
    for condition_name, excluded_rules in conditions.items():
        print(f"\n{'='*60}")
        print(f"Processing condition: {condition_name}")
        print(f"{'='*60}")
        
        # Generate segments for this condition
        segments_df = generate_segments_for_condition(all_data, condition_name, excluded_rules)
        
        # Save segments
        segments_file = output_dir / f"segments_{condition_name.lower()}.csv"
        segments_df.to_csv(segments_file, index=False)
        print(f"💾 Saved segments to {segments_file}")
        
        # Evaluate performance
        results = evaluate_performance_by_frames(segments_df, ground_truth_df)
        detailed_metrics = calculate_detailed_metrics(results)
        
        # Store results
        all_results[condition_name] = {
            'results': results,
            'detailed_metrics': detailed_metrics,
            'segments_df': segments_df
        }
        
        # Print summary
        overall_f1 = detailed_metrics.get('macro_avg', {}).get('f1_score', 0)
        print(f"📊 {condition_name}: Macro F1-Score = {overall_f1:.4f}")
    
    return all_results

def create_comprehensive_visualization(all_results, output_dir):
    """
    Create a focused visualization showing the F1 performance drop when each rule is excluded.
    
    Parameters
    ----------
    all_results : dict
        Results from all conditions
    output_dir : Path
        Output directory for plots
    """
    print("\n🎨 Creating rule impact visualization...")
    
    # Extract baseline performance (all rules)
    baseline_f1 = all_results['All_Rules']['detailed_metrics'].get('macro_avg', {}).get('f1_score', 0)
    
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
        colors.append(rule_info['color'])  # Use custom color
        
        # Get F1 score when this rule is excluded
        excluded_f1 = all_results[rule_info['condition']]['detailed_metrics'].get('macro_avg', {}).get('f1_score', 0)
        
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
    plot_file = output_dir / 'rule_impact_f1_drops.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Rule impact visualization saved to {plot_file}")
    
    # Print summary to console
    print(f"\n📊 RULE IMPACT ANALYSIS")
    print("="*50)
    print(f"Baseline F1-Score (All Rules): {baseline_f1:.4f}")
    print("-"*50)
    
    for rule_name, f1_drop in zip([r['rule_name'].replace('\n', ' ') for r in rule_data], f1_drops):
        percentage_drop = (f1_drop / baseline_f1) * 100 if baseline_f1 > 0 else 0
        impact_level = "HIGH" if f1_drop > 0.01 else "MODERATE" if f1_drop > 0.005 else "LOW"
        print(f"{rule_name:<25}: -{f1_drop:.4f} ({percentage_drop:.1f}%) [{impact_level}]")
    
    # Create summary table
    create_summary_table(all_results, output_dir)

def load_existing_results(output_dir):
    """
    Load existing analysis results from saved files.
    
    Parameters
    ----------
    output_dir : Path
        Directory containing saved results
        
    Returns
    -------
    dict or None
        Loaded results if successful, None otherwise
    """
    print("🔄 Loading existing analysis results...")
    
    # Check if summary file exists
    summary_file = output_dir / 'rule_ablation_summary.csv'
    if not summary_file.exists():
        print("❌ No existing results found. Run full analysis first.")
        return None
    
    try:
        # Load the summary data
        summary_df = pd.read_csv(summary_file)
        
        # Reconstruct results structure for visualization
        all_results = {}
        
        for _, row in summary_df.iterrows():
            condition_name = row['Condition']
            
            # Create mock detailed metrics structure
            detailed_metrics = {
                'macro_avg': {
                    'f1_score': row['Macro_F1'],
                    'precision': row['Macro_Precision'],
                    'recall': row['Macro_Recall']
                }
            }
            
            # Add class-specific metrics
            for class_name in ['interacting', 'available', 'alone']:
                detailed_metrics[class_name] = {
                    'f1_score': row[f'{class_name.capitalize()}_F1'],
                    'precision': row[f'{class_name.capitalize()}_Precision'],
                    'recall': row[f'{class_name.capitalize()}_Recall']
                }
            
            # Create mock results structure
            results = {
                'overall_accuracy': row['Overall_Accuracy']
            }
            
            all_results[condition_name] = {
                'results': results,
                'detailed_metrics': detailed_metrics
            }
        
        print(f"✅ Successfully loaded results for {len(all_results)} conditions")
        return all_results
        
    except Exception as e:
        print(f"❌ Error loading existing results: {e}")
        return None

def create_summary_table(all_results, output_dir):
    """Create a summary table with all metrics."""
    
    summary_data = []
    
    for condition_name, data in all_results.items():
        detailed_metrics = data['detailed_metrics']
        results = data['results']
        
        # Overall metrics
        macro_avg = detailed_metrics.get('macro_avg', {})
        
        row = {
            'Condition': condition_name,
            'Overall_Accuracy': results.get('overall_accuracy', 0),
            'Macro_F1': macro_avg.get('f1_score', 0),
            'Macro_Precision': macro_avg.get('precision', 0),
            'Macro_Recall': macro_avg.get('recall', 0),
        }
        
        # Class-specific F1 scores
        for class_name in ['interacting', 'available', 'alone']:
            if class_name in detailed_metrics:
                row[f'{class_name.capitalize()}_F1'] = detailed_metrics[class_name]['f1_score']
                row[f'{class_name.capitalize()}_Precision'] = detailed_metrics[class_name]['precision']
                row[f'{class_name.capitalize()}_Recall'] = detailed_metrics[class_name]['recall']
            else:
                row[f'{class_name.capitalize()}_F1'] = 0
                row[f'{class_name.capitalize()}_Precision'] = 0
                row[f'{class_name.capitalize()}_Recall'] = 0
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_file = output_dir / 'rule_ablation_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"📊 Summary table saved to {summary_file}")
    
    # Print to console
    print("\n" + "="*80)
    print("RULE ABLATION ANALYSIS SUMMARY")
    print("="*80)
    print(f"{'Condition':<25} {'Macro F1':<10} {'Interacting F1':<15} {'Available F1':<13} {'Alone F1':<10}")
    print("-"*80)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Condition']:<25} {row['Macro_F1']:<10.4f} {row['Interacting_F1']:<15.4f} {row['Available_F1']:<13.4f} {row['Alone_F1']:<10.4f}")

def main():
    """Main function to run the complete ablation analysis or regenerate plots."""
    
    parser = argparse.ArgumentParser(description='Rule Ablation Analysis for Social Interaction Classification')
    parser.add_argument('--db_path', type=str, default=str(DataPaths.INFERENCE_DB_PATH),
                       help='Path to the inference database')
    parser.add_argument('--plot_only', action='store_true',
                       help='Only regenerate plots from existing results (skip full analysis)')
    parser.add_argument('--output_dir', type=str, 
                       help='Custom output directory')
    
    args = parser.parse_args()
    
    # Setup paths
    db_path = Path(args.db_path)
    ground_truth_path = Inference.GROUND_TRUTH_SEGMENTS_CSV
    
    # Use custom output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Inference.BASE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.plot_only:
        print("🎨 Plot-only mode: Regenerating visualizations from existing results...")
        
        # Load existing results
        all_results = load_existing_results(output_dir)
        
        if all_results is None:
            print("❌ Cannot generate plots without existing results. Run full analysis first.")
            sys.exit(1)
        
        # Generate new plots
        create_comprehensive_visualization(all_results, output_dir)
        print(f"\n🎉 Plot regeneration complete! Visualization saved to {output_dir}")
        
    else:
        print("🚀 Full analysis mode: Running complete rule ablation analysis...")
        
        # Load ground truth
        print(f"📖 Loading ground truth from {ground_truth_path}")
        ground_truth_df = pd.read_csv(ground_truth_path, delimiter=';')
        
        # Run ablation analysis
        all_results = run_ablation_analysis(db_path, ground_truth_df, output_dir)
        
        # Create visualizations
        create_comprehensive_visualization(all_results, output_dir)
        
        print(f"\n🎉 Rule ablation analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()

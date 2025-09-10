#!/usr/bin/env python3
"""
Simple runner script for hyperparameter tuning.

This script provides an easy way to start the hyperparameter tuning process
with different configurations.

Usage:
    python run_hyperparameter_tuning.py [--quick] [--full]
    
Options:
    --quick: Run a quick test with just 5 combinations
    --full: Run the full hyperparameter search (can take hours)
    (default): Run with 20 combinations (balanced approach)
"""
import argparse
from tune_hyperparameters import main as tune_hyperparameters
from config import InferenceConfig

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning for social interaction analysis')
    parser.add_argument('--quick', action='store_true', 
                    help='Quick test with 5 combinations (~30 minutes)')
    parser.add_argument('--full', action='store_true',
                    help='Full search with all valid combinations (can take hours)')
    parser.add_argument('--max-combos', type=int, default=20,
                    help='Maximum number of combinations to test (default: 20)')
    
    args = parser.parse_args()
    
    # Determine number of combinations
    if args.quick:
        max_combinations = 5
        print("🚀 Running QUICK hyperparameter tuning (5 combinations)")
    elif args.full:
        max_combinations = None  # No limit
        print("🚀 Running FULL hyperparameter tuning (all valid combinations)")
    else:
        max_combinations = InferenceConfig.MAX_COMBINATIONS_TUNING
        print(f"🚀 Running hyperparameter tuning with {max_combinations} combinations")
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING FOR SOCIAL INTERACTION ANALYSIS")
    print("="*80)
    print("\nThis process will:")
    print("1. Generate combinations of hyperparameters")
    print("2. Run frame-level and video-level analysis for each combination")
    print("3. Evaluate performance against ground truth data")
    print("4. Identify the best performing configuration")
    
    # Run the hyperparameter tuning
    try:
        tune_hyperparameters()
        print("\n🎉 Hyperparameter tuning completed successfully!")
        print("\nCheck the 'hyperparameter_tuning_results' directory for:")
        print("  - best_configuration.json: Optimal hyperparameters")
        print("  - results_summary.csv: Performance of all combinations")
        print("  - Individual combo directories with detailed results")
        return 0
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        print("Partial results may be available in 'hyperparameter_tuning_results' directory")
        return 1
    except Exception as e:
        print(f"\n❌ Error during hyperparameter tuning: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

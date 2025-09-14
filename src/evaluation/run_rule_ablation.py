#!/usr/bin/env python3
"""
Runner script for Rule Ablation Analysis

This script provides a simple interface to run the comprehensive rule ablation analysis.
It will generate 5 different segment sets and evaluate their performance, or regenerate plots only.

Usage:
    python run_rule_ablation.py                    # Run full analysis
    python run_rule_ablation.py --plot_only        # Only regenerate plots
    python run_rule_ablation.py --output_dir custom_results
"""

import subprocess
import sys
from pathlib import Path

def run_rule_ablation():
    """Run the rule ablation analysis."""
    
    # Check if --plot_only flag is present
    plot_only_mode = '--plot_only' in sys.argv
    
    if plot_only_mode:
        print("🎨 Starting Plot Regeneration Mode")
        print("="*50)
    else:
        print("🚀 Starting Full Rule Ablation Analysis")
        print("="*50)
    
    # Get the script path
    script_path = Path(__file__).parent / "rule_ablation_analysis.py"
    
    if not script_path.exists():
        print(f"❌ Error: Script not found at {script_path}")
        return False
    
    try:
        # Run the analysis
        cmd = [sys.executable, str(script_path)]
        
        # Add all arguments (including --plot_only if present)
        if len(sys.argv) > 1:
            cmd.extend(sys.argv[1:])
        
        print(f"🔄 Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            if plot_only_mode:
                print("\n✅ Plot regeneration completed successfully!")
            else:
                print("\n✅ Rule ablation analysis completed successfully!")
            return True
        else:
            print(f"\n❌ Analysis failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return False

if __name__ == "__main__":
    success = run_rule_ablation()
    sys.exit(0 if success else 1)

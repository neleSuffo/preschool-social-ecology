#!/usr/bin/env python3
"""
GPU Training Launcher
Sets up CUDA library paths and launches training script
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_cuda_paths():
    """Set up CUDA library paths from virtual environment"""
    
    # Get virtual environment path
    venv_path = Path(sys.executable).parent.parent
    
    # CUDA library paths in the virtual environment
    cuda_lib_paths = [
        venv_path / "lib/python3.8/site-packages/nvidia/cublas/lib",
        venv_path / "lib/python3.8/site-packages/nvidia/cudnn/lib", 
        venv_path / "lib/python3.8/site-packages/nvidia/cuda_runtime/lib",
        venv_path / "lib/python3.8/site-packages/nvidia/cufft/lib",
        venv_path / "lib/python3.8/site-packages/nvidia/curand/lib",
        venv_path / "lib/python3.8/site-packages/nvidia/cusolver/lib",
        venv_path / "lib/python3.8/site-packages/nvidia/cusparse/lib",
    ]
    
    # Get existing LD_LIBRARY_PATH
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    
    # Add CUDA paths that exist
    new_paths = []
    for path in cuda_lib_paths:
        if path.exists():
            new_paths.append(str(path))
            print(f"✅ Found CUDA library path: {path}")
        else:
            print(f"⚠️ CUDA library path not found: {path}")
    
    if new_paths:
        # Combine new paths with existing LD_LIBRARY_PATH
        if current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths) + ':' + current_ld_path
        else:
            os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths)
        
        print(f"🔧 Set LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
        return True
    else:
        print("❌ No CUDA library paths found!")
        return False

def test_cuda_setup():
    """Test if CUDA setup is working"""
    print("🧪 Testing CUDA setup...")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
        
        # List GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU devices found: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
        
        if gpus:
            # Try a simple GPU operation
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([1.0, 2.0, 3.0])
                result = tf.reduce_sum(test_tensor)
                print("✅ GPU operation successful!")
                return True
        else:
            print("⚠️ No GPU devices found")
            return False
            
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 GPU Training Launcher")
    print("=" * 50)
    
    # Setup CUDA paths
    cuda_setup_success = setup_cuda_paths()
    
    if cuda_setup_success:
        # Test CUDA setup
        cuda_test_success = test_cuda_setup()
        
        if cuda_test_success:
            print("\n✅ CUDA setup successful! Launching training...")
        else:
            print("\n⚠️ CUDA test failed but proceeding anyway...")
            print("The training script will attempt to use available hardware.")
    else:
        print("\n⚠️ CUDA library paths not found. Training will use CPU.")
    
    # Launch the training script with any command line arguments
    if len(sys.argv) > 1:
        script_path = sys.argv[1]
        script_args = sys.argv[2:] if len(sys.argv) > 2 else []
        
        print(f"\n🏃 Launching: python {script_path} {' '.join(script_args)}")
        
        # Execute the training script
        try:
            subprocess.run([sys.executable, script_path] + script_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Training script failed with exit code {e.returncode}")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
            sys.exit(1)
    else:
        print("\n📝 Usage: python launch_gpu_training.py <training_script.py> [args...]")
        print("Example: python launch_gpu_training.py src/models/speech_type/train_audio_classifier.py")
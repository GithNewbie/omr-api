#!/usr/bin/env python3
"""
Smart installation script for Sheet Music API
Automatically detects Python version and installs compatible dependencies
"""

import sys
import subprocess
import platform

def get_python_version():
    """Get Python version as tuple (major, minor)"""
    return sys.version_info[:2]

def run_command(command, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🎵 Sheet Music API - Smart Installer")
    print("="*60)
    
    # Detect Python version
    py_version = get_python_version()
    py_version_str = f"{py_version[0]}.{py_version[1]}"
    
    print(f"✅ Python version detected: {py_version_str}")
    print(f"✅ Platform: {platform.system()} {platform.machine()}")
    
    # Check Python version compatibility
    if py_version < (3, 10):
        print(f"\n❌ Python {py_version_str} is not supported!")
        print("   Please use Python 3.10 or newer")
        sys.exit(1)
    
    if py_version >= (3, 13):
        print(f"\n⚠️  Warning: Python {py_version_str} is very new")
        print("   Some packages may not be available yet")
    
    # Step 1: Upgrade pip
    print("\n" + "="*60)
    print("Step 1/4: Upgrading pip, setuptools, and wheel")
    print("="*60)
    
    if not run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        "Upgrading pip tools"
    ):
        print("⚠️  Failed to upgrade pip, continuing anyway...")
    
    # Step 2: Select requirements file
    print("\n" + "="*60)
    print("Step 2/4: Selecting appropriate requirements file")
    print("="*60)
    
    if py_version >= (3, 12):
        requirements_file = "requirements-py312.txt"
        print(f"📄 Using: {requirements_file} (Python 3.12+)")
    elif py_version >= (3, 10):
        requirements_file = "requirements-py310-311.txt"
        print(f"📄 Using: {requirements_file} (Python 3.10-3.11)")
    else:
        requirements_file = "requirements.txt"
        print(f"📄 Using: {requirements_file} (fallback)")
    
    # Check if file exists, fallback to default
    try:
        with open(requirements_file, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"⚠️  {requirements_file} not found, using requirements.txt")
        requirements_file = "requirements.txt"
    
    # Step 3: Install core dependencies first
    print("\n" + "="*60)
    print("Step 3/4: Installing core dependencies")
    print("="*60)
    
    core_packages = [
        "numpy>=1.23.0,<2.0.0",
        "Pillow>=10.0.0",
    ]
    
    for package in core_packages:
        run_command(
            [sys.executable, "-m", "pip", "install", package],
            f"Installing {package.split('>=')[0]}"
        )
    
    # Step 4: Install all dependencies
    print("\n" + "="*60)
    print("Step 4/4: Installing all dependencies")
    print("="*60)
    
    if not run_command(
        [sys.executable, "-m", "pip", "install", "-r", requirements_file],
        f"Installing from {requirements_file}"
    ):
        print("\n❌ Installation failed!")
        print("\n🔧 Trying alternative installation method...")
        
        # Fallback: Install with loose constraints
        packages = [
            "fastapi",
            "uvicorn",
            "python-multipart",
            "opencv-python",
            "scipy",
            "pandas",
            "matplotlib",
            "torch",
            "torchvision",
            "ultralytics"
        ]
        
        for package in packages:
            run_command(
                [sys.executable, "-m", "pip", "install", package],
                f"Installing {package}"
            )
    
    # Verification
    print("\n" + "="*60)
    print("🔍 Verifying Installation")
    print("="*60)
    
    try:
        import cv2
        import numpy
        import ultralytics
        import fastapi
        import torch
        
        print("✅ All critical packages imported successfully!")
        print(f"   - OpenCV: {cv2.__version__}")
        print(f"   - NumPy: {numpy.__version__}")
        print(f"   - FastAPI: {fastapi.__version__}")
        print(f"   - PyTorch: {torch.__version__}")
        print(f"   - Ultralytics: {ultralytics.__version__}")
        
    except ImportError as e:
        print(f"⚠️  Warning: Could not import {e.name}")
        print("   Some packages may not be installed correctly")
        return False
    
    # Final instructions
    print("\n" + "="*60)
    print("🎉 Installation Complete!")
    print("="*60)
    print("\n📝 Next steps:")
    print("   1. Place your YOLO model at: runs/detect/clef_detector/weights/best.pt")
    print("   2. Start the server: python main.py")
    print("   3. Test the API: python test_api.py")
    print("   4. View docs at: http://localhost:8000/docs")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

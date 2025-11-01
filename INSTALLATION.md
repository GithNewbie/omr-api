# Installation Guide

This guide will help you set up the Sheet Music API without dependency conflicts.

## 🔍 Step 0: Check Your Python Version

```bash
python --version
# or
python3 --version
```

**Recommended**: Python 3.10, 3.11, or 3.12

---

## 📦 Installation Methods

### Method 1: Automatic Installation (Recommended)

Use the smart installation script that automatically detects your Python version:

#### Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies (auto-detects Python version)
python -m pip install -r requirements.txt
```

#### Linux/macOS (Bash)
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

---

### Method 2: Python Version-Specific Installation

If Method 1 fails, use the specific requirements file for your Python version:

#### For Python 3.12+
```bash
pip install -r requirements-py312.txt
```

#### For Python 3.10 or 3.11
```bash
pip install -r requirements-py310-311.txt
```

---

### Method 3: Manual Dependency Resolution

If you still encounter issues, install packages in this order:

```bash
# 1. Core dependencies first
pip install numpy>=1.23.0,<2.0.0
pip install Pillow>=10.0.0

# 2. Scientific packages
pip install scipy>=1.11.0
pip install pandas>=2.1.0
pip install matplotlib>=3.8.0

# 3. PyTorch (will auto-select compatible versions)
pip install torch torchvision

# 4. Computer vision
pip install opencv-python>=4.8.0

# 5. YOLO
pip install ultralytics>=8.0.0

# 6. Web framework
pip install fastapi uvicorn python-multipart
```

---

## 🐛 Troubleshooting Common Issues

### Issue 1: NumPy Version Conflicts

**Error**: `ERROR: Cannot install because these package versions have conflicting dependencies`

**Solution**:
```bash
# Uninstall all conflicting packages
pip uninstall -y numpy opencv-python ultralytics torch torchvision scipy pandas matplotlib

# Reinstall with loose constraints
pip install "numpy>=1.23.0,<2.0.0"
pip install opencv-python scipy pandas matplotlib
pip install torch torchvision
pip install ultralytics
```

### Issue 2: PyTorch Not Found for Your Python Version

**Error**: `ERROR: Could not find a version that satisfies the requirement torch==X.X.X`

**Solution**: Let pip find a compatible version automatically:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For CUDA support (NVIDIA GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 3: Ultralytics Python Version Error

**Error**: `ERROR: Ignored the following versions that require a different python version`

**Solution**: Use a compatible Ultralytics version:
```bash
# For Python 3.12+
pip install "ultralytics>=8.3.0"

# For Python 3.10-3.11
pip install "ultralytics>=8.0.0,<8.3.0"
```

### Issue 4: Microsoft Visual C++ Error (Windows)

**Error**: `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution**:
1. Download and install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Or use pre-built wheels:
   ```bash
   pip install --only-binary :all: opencv-python scipy
   ```

### Issue 5: pip is Out of Date

**Solution**:
```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## ✅ Verify Installation

After installation, verify everything works:

```bash
python -c "import cv2; import numpy; import ultralytics; import fastapi; print('All packages imported successfully!')"
```

---

## 🚀 Quick Start After Installation

1. **Place your YOLO model**:
   ```
   runs/detect/clef_detector/weights/best.pt
   ```

2. **Start the server**:
   ```bash
   python main.py
   ```

3. **Test the API**:
   - Open browser: http://localhost:8000/docs
   - Or run: `python test_api.py`

---

## 🔄 Clean Installation (Nuclear Option)

If nothing works, start completely fresh:

### Windows
```powershell
# Delete old virtual environment
Remove-Item -Recurse -Force venv

# Create new one
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install with no cache
pip install --no-cache-dir -r requirements.txt
```

### Linux/macOS
```bash
# Delete old virtual environment
rm -rf venv

# Create new one
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install with no cache
pip install --no-cache-dir -r requirements.txt
```

---

## 📋 Tested Configurations

| Python | OS | Status | Notes |
|--------|----|---------|----|
| 3.12 | Windows 11 | ✅ Working | Use requirements-py312.txt |
| 3.12 | Ubuntu 22.04 | ✅ Working | Use requirements-py312.txt |
| 3.11 | Windows 10/11 | ✅ Working | Use requirements.txt |
| 3.11 | macOS | ✅ Working | Use requirements.txt |
| 3.10 | Ubuntu 20.04 | ✅ Working | Use requirements-py310-311.txt |
| 3.9 | Any | ⚠️ Limited | Some packages may not support |

---

## 🆘 Still Having Issues?

1. **Check Python version**: Make sure you're using Python 3.10+
   ```bash
   python --version
   ```

2. **Use a virtual environment**: Always use venv to avoid system conflicts

3. **Update pip**: Old pip versions cause many issues
   ```bash
   python -m pip install --upgrade pip
   ```

4. **Check for system packages**: On Linux, install system dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install python3-dev python3-pip
   sudo apt-get install libgl1-mesa-glx libglib2.0-0
   ```

5. **Try CPU-only PyTorch**: Smaller and fewer dependencies
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

---

## 💡 Pro Tips

### Tip 1: Lock Your Dependencies
After successful installation, freeze your working versions:
```bash
pip freeze > requirements-working.txt
```

### Tip 2: Use pyenv for Multiple Python Versions
```bash
# Install pyenv (Linux/macOS)
curl https://pyenv.run | bash

# Install Python 3.11
pyenv install 3.11.0
pyenv local 3.11.0
```

### Tip 3: Docker for Consistent Environment
If you keep having issues, use Docker:
```bash
docker-compose up -d
```

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [OpenCV Installation Issues](https://github.com/opencv/opencv-python/issues)

---

**Need more help?** Check the project's GitHub issues or create a new one with:
- Your Python version
- Operating system
- Full error message
- Output of `pip list`

# Installation Quick Reference

## 🚀 One-Command Install

### Windows
```powershell
.\install.ps1
```

### Linux/macOS
```bash
chmod +x install.sh && ./install.sh
```

### Python (Any OS)
```bash
python install.py
```

---

## 🔧 Common Issues & Fixes

### ❌ NumPy Conflict Error
```bash
pip uninstall -y numpy opencv-python ultralytics torch scipy pandas matplotlib
pip install "numpy>=1.23.0,<2.0.0"
pip install -r requirements.txt
```

### ❌ PyTorch Not Found
```bash
# CPU-only (smaller, faster)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### ❌ Python Version Too Old
**Minimum**: Python 3.10

**Check version:**
```bash
python --version
```

**Download:** https://www.python.org/downloads/

### ❌ Microsoft Visual C++ Error (Windows)
Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/

Or use pre-built wheels:
```bash
pip install --only-binary :all: opencv-python scipy
```

### ❌ Permission Denied (Linux/macOS)
```bash
sudo apt-get install python3-dev python3-pip
# or
brew install python
```

---

## 📋 Requirements Files

| File | Use When |
|------|----------|
| `requirements.txt` | Auto-detect (default) |
| `requirements-py312.txt` | Python 3.12+ |
| `requirements-py310-311.txt` | Python 3.10-3.11 |

---

## ✅ Verify Installation

```bash
python -c "import cv2, numpy, ultralytics, fastapi, torch; print('OK')"
```

---

## 🔄 Start Fresh

### Windows
```powershell
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🎯 After Installation

1. **Add YOLO model:**
   ```
   runs/detect/clef_detector/weights/best.pt
   ```

2. **Start server:**
   ```bash
   python main.py
   ```

3. **Test API:**
   ```bash
   python test_api.py
   ```

4. **View docs:**
   ```
   http://localhost:8000/docs
   ```

---

## 💡 Pro Tips

- Always use a virtual environment
- Update pip first: `python -m pip install --upgrade pip`
- Use Python 3.11 for best compatibility
- Lock working versions: `pip freeze > requirements-working.txt`

---

## 📞 Need Help?

1. Read [INSTALLATION.md](INSTALLATION.md) - detailed guide
2. Check [README.md](README.md) - full documentation  
3. Run `python install.py` - smart installer with auto-fixes

---

## 🐳 Docker Alternative

If nothing works, use Docker:

```bash
docker-compose up -d
```

No Python setup needed!

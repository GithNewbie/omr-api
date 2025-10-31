#!/bin/bash
# Sheet Music API - Linux/macOS Installation Script

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🎵 Sheet Music API - Unix Installer${NC}"
echo -e "${CYAN}====================================${NC}"

# Check Python installation
echo -e "\n${YELLOW}📍 Step 1: Checking Python installation...${NC}"

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd --version 2>&1)
        PYTHON_CMD=$cmd
        echo -e "${GREEN}✅ Found: $version using '$cmd'${NC}"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}❌ Python not found! Please install Python 3.10+${NC}"
    exit 1
fi

# Get Python version
VERSION_OUTPUT=$($PYTHON_CMD --version 2>&1)
if [[ $VERSION_OUTPUT =~ Python[[:space:]]([0-9]+)\.([0-9]+) ]]; then
    MAJOR_VERSION=${BASH_REMATCH[1]}
    MINOR_VERSION=${BASH_REMATCH[2]}
    
    if [ "$MAJOR_VERSION" -lt 3 ] || ([ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -lt 10 ]); then
        echo -e "${RED}❌ Python $MAJOR_VERSION.$MINOR_VERSION is too old! Please use Python 3.10+${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Python version is compatible: $MAJOR_VERSION.$MINOR_VERSION${NC}"
fi

# Create virtual environment
echo -e "\n${YELLOW}📍 Step 2: Creating virtual environment...${NC}"

if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}🗑️  Removing old virtual environment...${NC}"
        rm -rf venv
    else
        echo -e "${CYAN}ℹ️  Using existing virtual environment${NC}"
    fi
fi

if [ ! -d "venv" ]; then
    echo -e "${CYAN}📦 Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv venv
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to create virtual environment!${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}📍 Step 3: Activating virtual environment...${NC}"

source venv/bin/activate

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to activate virtual environment!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Upgrade pip
echo -e "\n${YELLOW}📍 Step 4: Upgrading pip...${NC}"
python -m pip install --upgrade pip setuptools wheel

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Warning: Failed to upgrade pip${NC}"
fi

# Select requirements file based on Python version
echo -e "\n${YELLOW}📍 Step 5: Selecting requirements file...${NC}"

REQUIREMENTS_FILE="requirements.txt"
if [ "$MINOR_VERSION" -ge 12 ]; then
    REQUIREMENTS_FILE="requirements-py312.txt"
    echo -e "${CYAN}📄 Using: $REQUIREMENTS_FILE (Python 3.12+)${NC}"
elif [ "$MINOR_VERSION" -ge 10 ]; then
    REQUIREMENTS_FILE="requirements-py310-311.txt"
    echo -e "${CYAN}📄 Using: $REQUIREMENTS_FILE (Python 3.10-3.11)${NC}"
fi

# Fallback to default if specific file doesn't exist
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${YELLOW}⚠️  $REQUIREMENTS_FILE not found, using requirements.txt${NC}"
    REQUIREMENTS_FILE="requirements.txt"
fi

# Install dependencies
echo -e "\n${YELLOW}📍 Step 6: Installing dependencies...${NC}"
echo -e "${CYAN}This may take several minutes...${NC}"

pip install -r $REQUIREMENTS_FILE

if [ $? -ne 0 ]; then
    echo -e "\n${YELLOW}⚠️  Installation encountered errors${NC}"
    echo -e "${CYAN}Trying alternative installation method...${NC}"
    
    # Try installing packages individually
    packages=(
        "numpy>=1.23.0,<2.0.0"
        "Pillow>=10.0.0"
        "fastapi"
        "uvicorn"
        "python-multipart"
        "opencv-python"
        "scipy"
        "pandas"
        "matplotlib"
        "torch"
        "torchvision"
        "ultralytics"
    )
    
    for package in "${packages[@]}"; do
        echo -e "${CYAN}Installing $package...${NC}"
        pip install "$package"
    done
fi

# Verify installation
echo -e "\n${YELLOW}📍 Step 7: Verifying installation...${NC}"

python << EOF
import sys
try:
    import cv2
    import numpy
    import ultralytics
    import fastapi
    import torch
    print('✅ All packages imported successfully!')
    print(f'   OpenCV: {cv2.__version__}')
    print(f'   NumPy: {numpy.__version__}')
    print(f'   FastAPI: {fastapi.__version__}')
    print(f'   PyTorch: {torch.__version__}')
    print(f'   Ultralytics: {ultralytics.__version__}')
    sys.exit(0)
except ImportError as e:
    print(f'❌ Error: Could not import {e.name}')
    sys.exit(1)
EOF

# Final instructions
echo -e "\n${GREEN}🎉 Installation Complete!${NC}"
echo -e "${GREEN}=========================${NC}"
echo -e "\n${CYAN}📝 Next steps:${NC}"
echo "   1. Place your YOLO model at: runs/detect/clef_detector/weights/best.pt"
echo "   2. Start the server: python main.py"
echo "   3. Test the API: python test_api.py"
echo "   4. View docs at: http://localhost:8000/docs"
echo -e "\n${CYAN}💡 To activate this environment in the future, run:${NC}"
echo -e "${YELLOW}   source venv/bin/activate${NC}"
echo ""

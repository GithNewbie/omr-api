# Sheet Music API - Windows Installation Script
# Run this script in PowerShell

Write-Host "🎵 Sheet Music API - Windows Installer" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Check Python installation
Write-Host "`n📍 Step 1: Checking Python installation..." -ForegroundColor Yellow

$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $cmd
            Write-Host "✅ Found: $version using '$cmd'" -ForegroundColor Green
            break
        }
    } catch {}
}

if (-not $pythonCmd) {
    Write-Host "❌ Python not found! Please install Python 3.10+ from https://www.python.org/" -ForegroundColor Red
    exit 1
}

# Get Python version
$versionOutput = & $pythonCmd --version 2>&1
if ($versionOutput -match "Python (\d+)\.(\d+)") {
    $majorVersion = [int]$matches[1]
    $minorVersion = [int]$matches[2]
    
    if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 10)) {
        Write-Host "❌ Python $majorVersion.$minorVersion is too old! Please use Python 3.10+" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ Python version is compatible: $majorVersion.$minorVersion" -ForegroundColor Green
}

# Create virtual environment
Write-Host "`n📍 Step 2: Creating virtual environment..." -ForegroundColor Yellow

if (Test-Path "venv") {
    Write-Host "⚠️  Virtual environment already exists" -ForegroundColor Yellow
    $response = Read-Host "Do you want to recreate it? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "🗑️  Removing old virtual environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force venv
    } else {
        Write-Host "ℹ️  Using existing virtual environment" -ForegroundColor Cyan
    }
}

if (-not (Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Cyan
    & $pythonCmd -m venv venv
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to create virtual environment!" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`n📍 Step 3: Activating virtual environment..." -ForegroundColor Yellow

$activateScript = "venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "🔄 Activating virtual environment..." -ForegroundColor Cyan
    & $activateScript
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "❌ Activation script not found!" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "`n📍 Step 4: Upgrading pip..." -ForegroundColor Yellow
& python -m pip install --upgrade pip setuptools wheel

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Warning: Failed to upgrade pip" -ForegroundColor Yellow
}

# Select requirements file based on Python version
Write-Host "`n📍 Step 5: Selecting requirements file..." -ForegroundColor Yellow

$requirementsFile = "requirements.txt"
if ($minorVersion -ge 12) {
    $requirementsFile = "requirements-py312.txt"
    Write-Host "📄 Using: $requirementsFile (Python 3.12+)" -ForegroundColor Cyan
} elseif ($minorVersion -ge 10) {
    $requirementsFile = "requirements-py310-311.txt"
    Write-Host "📄 Using: $requirementsFile (Python 3.10-3.11)" -ForegroundColor Cyan
}

# Fallback to default if specific file doesn't exist
if (-not (Test-Path $requirementsFile)) {
    Write-Host "⚠️  $requirementsFile not found, using requirements.txt" -ForegroundColor Yellow
    $requirementsFile = "requirements.txt"
}

# Install dependencies
Write-Host "`n📍 Step 6: Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Cyan

& pip install -r $requirementsFile

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n⚠️  Installation encountered errors" -ForegroundColor Yellow
    Write-Host "Trying alternative installation method..." -ForegroundColor Cyan
    
    # Try installing packages individually
    $packages = @(
        "numpy>=1.23.0,<2.0.0",
        "Pillow>=10.0.0",
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
    )
    
    foreach ($package in $packages) {
        Write-Host "Installing $package..." -ForegroundColor Cyan
        & pip install $package
    }
}

# Verify installation
Write-Host "`n📍 Step 7: Verifying installation..." -ForegroundColor Yellow

$verifyScript = @"
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
"@

$verifyResult = & python -c $verifyScript
Write-Host $verifyResult

# Final instructions
Write-Host "`n" -NoNewline
Write-Host "🎉 Installation Complete!" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green
Write-Host "`n📝 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Place your YOLO model at: runs\detect\clef_detector\weights\best.pt"
Write-Host "   2. Start the server: python main.py"
Write-Host "   3. Test the API: python test_api.py"
Write-Host "   4. View docs at: http://localhost:8000/docs"
Write-Host "`n💡 To activate this environment in the future, run:"
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""

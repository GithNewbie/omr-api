# Sheet Music to Audio API

A FastAPI application that converts sheet music images into audio files (WAV format). The system detects musical notes, clefs, and staff lines, then generates corresponding audio.

## Features

- 🎵 Converts sheet music images to audio (WAV)
- 🎼 Detects G-clef and F-clef using YOLOv8
- 📊 Identifies staff lines and musical notes
- 🎹 Supports different note types (black/white notes, stems, hooks)
- 🔊 Customizable audio parameters (duration, amplitude, pause)
- 📝 Returns detailed note information (optional)

## Installation

### Quick Install (Recommended)

#### Windows (PowerShell)
```powershell
# Run the automated installer
.\install.ps1
```

#### Linux/macOS (Bash)
```bash
# Make script executable
chmod +x install.sh

# Run the automated installer
./install.sh
```

#### Using Python Script (Cross-platform)
```bash
python install.py
```

### Manual Installation

If the automated installers don't work, follow these steps:

#### 1. Create a virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

#### 2. Upgrade pip
```bash
python -m pip install --upgrade pip setuptools wheel
```

#### 3. Install dependencies

Choose based on your Python version:

**Python 3.12+:**
```bash
pip install -r requirements-py312.txt
```

**Python 3.10-3.11:**
```bash
pip install -r requirements-py310-311.txt
```

**Auto-detect (recommended):**
```bash
pip install -r requirements.txt
```

#### 4. Setup YOLO Model

Place your trained YOLO model at:
```
runs/detect/clef_detector/weights/best.pt
```

Or update the `MODEL_PATH` in `main.py` to point to your model location.

### Troubleshooting

If you encounter dependency conflicts, see [INSTALLATION.md](INSTALLATION.md) for detailed troubleshooting steps.

## Running the API

### Start the server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check

```bash
GET /health
```

Check if the API is running and the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Process Sheet Music (Get Audio)

```bash
POST /process
```

Upload a sheet music image and receive a WAV audio file.

**Parameters:**
- `file` (required): Image file (PNG, JPG, etc.)
- `note_duration` (optional): Duration of each note in seconds (default: 0.5)
- `pause_duration` (optional): Pause between notes in seconds (default: 0.05)
- `amplitude` (optional): Sound amplitude (default: 0.5)
- `confidence` (optional): YOLO detection confidence (default: 0.05)

**Example using cURL:**

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@sheet_sample_4.png" \
  -F "note_duration=0.5" \
  -F "pause_duration=0.05" \
  -F "amplitude=0.5" \
  -o output.wav
```

**Example using Python requests:**

```python
import requests

url = "http://localhost:8000/process"
files = {"file": open("sheet_sample_4.png", "rb")}
params = {
    "note_duration": 0.5,
    "pause_duration": 0.05,
    "amplitude": 0.5
}

response = requests.post(url, files=files, params=params)

with open("output.wav", "wb") as f:
    f.write(response.content)

print("Audio file saved as output.wav")
```

### 3. Process with Detailed Information

```bash
POST /process_with_info
```

Upload a sheet music image and receive detailed note information in JSON format.

**Parameters:** Same as `/process`

**Response:**
```json
{
  "note_count": 15,
  "staff_count": 2,
  "notes": [
    {
      "id": 1,
      "staff_id": 0,
      "note_name": "E4",
      "type": "đen",
      "stem": "có đuôi",
      "hook": "móc đơn",
      "center_x": 150.5,
      "center_y": 200.3
    },
    ...
  ],
  "audio_file": "Use /process endpoint to download audio file",
  "message": "Processing completed successfully"
}
```

**Example using cURL:**

```bash
curl -X POST "http://localhost:8000/process_with_info" \
  -F "file=@sheet_sample_4.png" \
  | jq
```

**Example using Python requests:**

```python
import requests

url = "http://localhost:8000/process_with_info"
files = {"file": open("sheet_sample_4.png", "rb")}

response = requests.post(url, files=files)
data = response.json()

print(f"Detected {data['note_count']} notes across {data['staff_count']} staves")
for note in data['notes']:
    print(f"Note {note['id']}: {note['note_name']} ({note['type']}, {note['stem']})")
```

## Project Structure

```
sheet-music-api/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── test_api.py            # API testing script
└── runs/
    └── detect/
        └── clef_detector/
            └── weights/
                └── best.pt  # YOLO model weights
```

## Testing

Use the provided test script:

```bash
python test_api.py
```

Or test manually with your own images.

## Technical Details

### Image Processing Pipeline

1. **Image Loading**: Read and convert to grayscale
2. **Binarization**: Apply threshold to create binary image
3. **Staff Detection**: Detect horizontal staff lines using morphological operations
4. **Clef Detection**: Use YOLOv8 to detect G-clef and F-clef
5. **Note Detection**: Identify elliptical shapes (note heads)
6. **Note Classification**: Classify notes by type (black/white), stems, and hooks
7. **Note Mapping**: Map note positions to musical notes (A3-A5)
8. **Audio Generation**: Generate sine wave audio for each note

### Note Types

- **Đen (Black)**: Quarter notes or shorter
- **Trắng (White)**: Half notes or longer
- **Có đuôi (With stem)**: Notes with stems
- **Móc đơn (Single hook)**: Eighth notes
- **Móc kép (Double hook)**: Sixteenth notes

### Audio Generation

- Sample rate: 44,100 Hz
- Fade in/out: 0.01 seconds
- Note frequencies: A3 (220 Hz) to A5 (880 Hz)
- Output format: WAV (16-bit PCM)

## Troubleshooting

### Model not loading
- Ensure YOLO model path is correct in `main.py`
- Check that model file exists at specified path

### Poor detection results
- Adjust `confidence` parameter (try values between 0.01 and 0.1)
- Ensure input image is clear and high resolution
- Verify staff lines are horizontal and clearly visible

### Audio quality issues
- Adjust `note_duration` and `pause_duration`
- Modify `amplitude` for volume control
- Check that notes are correctly detected (use `/process_with_info`)

## Requirements

- Python 3.8+
- YOLO model trained on sheet music symbols
- Sufficient RAM for image processing (recommended: 4GB+)

## License

This project contains code for educational purposes.

## Contributing

Feel free to submit issues and enhancement requests!

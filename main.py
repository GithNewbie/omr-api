from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.io.wavfile import write
import os
import tempfile
import shutil
from typing import List, Dict, Any
import pandas as pd
from ultralytics import YOLO

app = FastAPI(title="Sheet Music to Audio API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model at startup
MODEL_PATH = 'runs/detect/clef_detector/weights/best.pt'
CLASS_NAMES = ['G-clef', 'F-clef']

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load YOLO model: {e}")
    model = None


@app.get("/")
async def root():
    return {
        "message": "Sheet Music to Audio API",
        "version": "1.0.0",
        "endpoints": {
            "/process": {
                "method": "POST",
                "description": "Upload sheet music image and get audio file for download",
                "returns": "WAV audio file"
            },
            "/process_with_info": {
                "method": "POST",
                "description": "Upload sheet music image and get detailed note information (JSON)",
                "returns": "JSON with note details"
            },
            "/process_with_audio_player": {
                "method": "POST",
                "description": "Upload sheet music and play audio directly in browser (NEW!)",
                "returns": "HTML page with embedded audio player",
                "note": "Perfect for testing in Swagger UI - audio plays immediately"
            },
            "/process_get_info": {
                "method": "POST",
                "description": "Get complete JSON info with base64 audio preview (NEW!)",
                "returns": "JSON with notes data + embedded audio",
                "note": "Shows exact /process_with_info output plus playable audio"
            },
            "/health": {
                "method": "GET",
                "description": "Check API health and model status",
                "returns": "Health status JSON"
            },
            "/docs": {
                "method": "GET",
                "description": "Interactive API documentation (Swagger UI)",
                "returns": "Web interface"
            }
        },
        "quick_start": {
            "1": "Go to http://localhost:8000/docs",
            "2": "Try /process_with_audio_player to hear your music instantly",
            "3": "Use /process to download the WAV file",
            "4": "Use /process_get_info to get JSON data with audio preview"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/process")
async def process_sheet_music(
    file: UploadFile = File(...),
    note_duration: float = 0.5,
    pause_duration: float = 0.05,
    amplitude: float = 0.5,
    confidence: float = 0.05
):
    """
    Process sheet music image and convert to audio
    
    Parameters:
    - file: Image file (PNG, JPG, etc.)
    - note_duration: Duration of each note in seconds (default: 0.5)
    - pause_duration: Pause between notes in seconds (default: 0.05)
    - amplitude: Sound amplitude (default: 0.5)
    - confidence: YOLO detection confidence (default: 0.05)
    
    Returns:
    - WAV audio file
    - JSON with note information
    """
    
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded file
        input_path = os.path.join(temp_dir, "input_image.png")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the image
        result = process_image(
            input_path,
            model,
            CLASS_NAMES,
            note_duration,
            pause_duration,
            amplitude,
            confidence,
            temp_dir
        )
        
        # Return the audio file
        return FileResponse(
            result["audio_path"],
            media_type="audio/wav",
            filename="sheet_music_output.wav",
            headers={
                "X-Note-Count": str(result["note_count"]),
                "X-Staff-Count": str(result["staff_count"])
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup will happen after response is sent
        pass


@app.post("/process_with_info")
async def process_sheet_music_with_info(
    file: UploadFile = File(...),
    note_duration: float = 0.5,
    pause_duration: float = 0.05,
    amplitude: float = 0.5,
    confidence: float = 0.05
):
    """
    Process sheet music image and return detailed information along with audio
    
    Returns:
    - JSON with notes information and audio file path
    """
    
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded file
        input_path = os.path.join(temp_dir, "input_image.png")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the image
        result = process_image(
            input_path,
            model,
            CLASS_NAMES,
            note_duration,
            pause_duration,
            amplitude,
            confidence,
            temp_dir
        )
        
        # Return detailed information
        return JSONResponse({
            "note_count": result["note_count"],
            "staff_count": result["staff_count"],
            "notes": result["notes_info"],
            "audio_file": "Use /process endpoint to download audio file",
            "message": "Processing completed successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@app.post("/process_with_audio_player")
async def process_sheet_music_with_audio_player(
    file: UploadFile = File(...),
    note_duration: float = 0.5,
    pause_duration: float = 0.05,
    amplitude: float = 0.5,
    confidence: float = 0.05
):
    """
    Process sheet music image and return HTML page with embedded audio player
    
    This endpoint allows you to play the audio directly in the browser without downloading.
    Perfect for testing in the Swagger UI at /docs
    
    Parameters:
    - file: Image file (PNG, JPG, etc.)
    - note_duration: Duration of each note in seconds (default: 0.5)
    - pause_duration: Pause between notes in seconds (default: 0.05)
    - amplitude: Sound amplitude (default: 0.5)
    - confidence: YOLO detection confidence (default: 0.05)
    
    Returns:
    - HTML page with audio player and note information
    """
    from fastapi.responses import HTMLResponse
    import base64
    
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded file
        input_path = os.path.join(temp_dir, "input_image.png")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the image
        result = process_image(
            input_path,
            model,
            CLASS_NAMES,
            note_duration,
            pause_duration,
            amplitude,
            confidence,
            temp_dir
        )
        
        # Read the audio file and encode as base64
        with open(result["audio_path"], "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Create HTML with embedded audio player
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sheet Music Audio Player</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #333;
                }}
                .container {{
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                }}
                h1 {{
                    color: #667eea;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .audio-section {{
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                audio {{
                    width: 100%;
                    max-width: 600px;
                    margin: 20px auto;
                    display: block;
                }}
                .info-box {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .info-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .info-card h3 {{
                    margin: 0 0 10px 0;
                    font-size: 14px;
                    opacity: 0.9;
                }}
                .info-card p {{
                    margin: 0;
                    font-size: 32px;
                    font-weight: bold;
                }}
                .notes-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                    background: white;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .notes-table th {{
                    background: #667eea;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                .notes-table td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #eee;
                }}
                .notes-table tr:hover {{
                    background: #f8f9fa;
                }}
                .download-btn {{
                    display: inline-block;
                    background: #667eea;
                    color: white;
                    padding: 12px 24px;
                    border-radius: 5px;
                    text-decoration: none;
                    margin: 10px;
                    transition: background 0.3s;
                }}
                .download-btn:hover {{
                    background: #764ba2;
                }}
                .note-type-black {{ color: #000; font-weight: bold; }}
                .note-type-white {{ color: #666; }}
                .success-badge {{
                    background: #28a745;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 20px;
                    display: inline-block;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎵 Sheet Music Audio Player</h1>
                
                <div class="audio-section">
                    <span class="success-badge">✓ Processing Complete</span>
                    <h2>🔊 Play Your Music</h2>
                    <audio controls autoplay>
                        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    <p style="margin-top: 15px; color: #666;">
                        <small>Audio is embedded in this page. Click play to listen!</small>
                    </p>
                </div>
                
                <div class="info-box">
                    <div class="info-card">
                        <h3>Total Notes</h3>
                        <p>{result["note_count"]}</p>
                    </div>
                    <div class="info-card">
                        <h3>Staff Lines</h3>
                        <p>{result["staff_count"]}</p>
                    </div>
                    <div class="info-card">
                        <h3>Duration</h3>
                        <p>{note_duration}s</p>
                    </div>
                    <div class="info-card">
                        <h3>Amplitude</h3>
                        <p>{amplitude}</p>
                    </div>
                </div>
                
                <h2>📋 Detected Notes</h2>
                <div style="overflow-x: auto;">
                    <table class="notes-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Staff</th>
                                <th>Note</th>
                                <th>Type</th>
                                <th>Stem</th>
                                <th>Hook</th>
                                <th>Position X</th>
                                <th>Position Y</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        # Add note rows
        for note in result["notes_info"][:50]:  # Show first 50 notes
            note_type_class = "note-type-black" if note["type"] == "đen" else "note-type-white"
            html_content += f"""
                            <tr>
                                <td>{note["id"]}</td>
                                <td>Staff {note["staff_id"]}</td>
                                <td><strong>{note["note_name"]}</strong></td>
                                <td class="{note_type_class}">{note["type"]}</td>
                                <td>{note["stem"]}</td>
                                <td>{note["hook"] if note["hook"] else "-"}</td>
                                <td>{note["center_x"]:.1f}</td>
                                <td>{note["center_y"]:.1f}</td>
                            </tr>
            """
        
        if len(result["notes_info"]) > 50:
            html_content += f"""
                            <tr>
                                <td colspan="8" style="text-align: center; color: #666; font-style: italic;">
                                    ... and {len(result["notes_info"]) - 50} more notes
                                </td>
                            </tr>
            """
        
        html_content += """
                        </tbody>
                    </table>
                </div>
                
                <div style="text-align: center; margin-top: 30px;">
                    <p style="color: #666;">
                        <small>💡 Tip: You can also use the <code>/process</code> endpoint to download the WAV file</small>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary files (keep for a moment to allow audio playback)
        pass


@app.post("/process_get_info")
async def process_get_complete_info(
    file: UploadFile = File(...),
    note_duration: float = 0.5,
    pause_duration: float = 0.05,
    amplitude: float = 0.5,
    confidence: float = 0.05
):
    """
    Process sheet music and return complete JSON information with audio preview
    
    This endpoint processes the image and returns the full JSON data that shows
    exactly what the /process_with_info endpoint would return, but also includes
    a base64-encoded audio preview that you can play directly.
    
    Perfect for:
    - Testing the processing pipeline
    - Getting structured data for further processing
    - Previewing results before downloading
    
    Parameters:
    - file: Image file (PNG, JPG, etc.)
    - note_duration: Duration of each note in seconds (default: 0.5)
    - pause_duration: Pause between notes in seconds (default: 0.05)
    - amplitude: Sound amplitude (default: 0.5)
    - confidence: YOLO detection confidence (default: 0.05)
    
    Returns:
    - JSON with complete note information and embedded audio data
    """
    import base64
    
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded file
        input_path = os.path.join(temp_dir, "input_image.png")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the image
        result = process_image(
            input_path,
            model,
            CLASS_NAMES,
            note_duration,
            pause_duration,
            amplitude,
            confidence,
            temp_dir
        )
        
        # Read the audio file and encode as base64
        with open(result["audio_path"], "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Return detailed information with audio
        return JSONResponse({
            "status": "success",
            "message": "Processing completed successfully",
            "processing_info": {
                "note_duration": note_duration,
                "pause_duration": pause_duration,
                "amplitude": amplitude,
                "confidence": confidence
            },
            "results": {
                "note_count": result["note_count"],
                "staff_count": result["staff_count"],
                "notes": result["notes_info"]
            },
            "audio": {
                "format": "wav",
                "encoding": "base64",
                "data": audio_base64,
                "size_bytes": len(audio_data),
                "duration_seconds": len(audio_data) / (44100 * 2),  # Approximate
                "sample_rate": 44100,
                "channels": 1,
                "bit_depth": 16,
                "usage_instructions": {
                    "html": f'<audio controls><source src="data:audio/wav;base64,{audio_base64[:50]}..." type="audio/wav"></audio>',
                    "javascript": "const audio = new Audio('data:audio/wav;base64,' + audioData); audio.play();",
                    "python": "import base64; audio_bytes = base64.b64decode(response['audio']['data'])"
                }
            },
            "api_info": {
                "endpoint": "/process_get_info",
                "compatible_with": "/process_with_info",
                "download_endpoint": "/process",
                "play_in_browser": "/process_with_audio_player"
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def process_image(
    img_path: str,
    model: YOLO,
    class_names: List[str],
    note_duration: float,
    pause_duration: float,
    amplitude: float,
    confidence: float,
    output_dir: str
) -> Dict[str, Any]:
    """Main processing function that contains all the original logic"""
    
    # Read image
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 1))
    detected_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    detected_horizontal = cv2.dilate(detected_horizontal, np.ones((3, 1), np.uint8), iterations=1)
    
    # Detect staff lines
    horizontal_sum = np.sum(detected_horizontal, axis=1)
    line_y_positions = np.where(horizontal_sum > np.max(horizontal_sum) * 0.5)[0]
    
    staff_lines_groups = []
    if len(line_y_positions) > 0:
        Z = sch.fclusterdata(line_y_positions.reshape(-1, 1), t=10, criterion='distance')
        for staff_id in np.unique(Z):
            staff_lines = sorted(line_y_positions[Z == staff_id])
            if len(staff_lines) >= 5:
                staff_lines_groups.append(staff_lines)
    
    # Clean staff lines
    staff_lines_groups_clean = []
    for staff in staff_lines_groups:
        staff = np.array(staff)
        clean_lines = []
        current_cluster = [staff[0]]
        
        for y in staff[1:]:
            if y - current_cluster[-1] <= 2:
                current_cluster.append(y)
            else:
                clean_lines.append(int(np.mean(current_cluster)))
                current_cluster = [y]
        clean_lines.append(int(np.mean(current_cluster)))
        
        if len(clean_lines) > 5:
            indices = np.linspace(0, len(clean_lines) - 1, 5, dtype=int)
            clean_lines = [clean_lines[i] for i in indices]
        
        staff_lines_groups_clean.append(clean_lines)
    
    staff_lines_groups = staff_lines_groups_clean
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    detected_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Find stem regions
    contours, _ = cv2.findContours(detected_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_stem = []
    for i, cnt in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        regions_stem.append({
            "id": i,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "contour": cnt
        })
    
    # Remove lines
    removed_lines = cv2.add(detected_horizontal, detected_vertical)
    only_notes = cv2.subtract(binary, removed_lines)
    
    # Repair notes
    kernel_repair = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    repaired = cv2.morphologyEx(only_notes, cv2.MORPH_CLOSE, kernel_repair, iterations=3)
    repaired = cv2.dilate(repaired, np.ones((2, 2), np.uint8), iterations=1)
    
    # Detect G-clef boxes
    g_clef_boxes = detect_g_clef_boxes_for_binary(model, img_path, class_names, confidence)
    
    # Remove G-clef from binary
    img_before_detect_note = repaired.copy()
    for x1, y1, x2, y2 in g_clef_boxes:
        img_before_detect_note[y1:y2, x1:x2] = 0
    
    if len(g_clef_boxes) > 0:
        for i, (x1, y1, x2, y2) in enumerate(g_clef_boxes, start=1):
            x_left = max(0, x1)
            img_before_detect_note[y1:y2, 0:x_left] = 0
        
        x_right_min = min(x2 for (x1, y1, x2, y2) in g_clef_boxes)
        img_before_detect_note[:, 0:x_right_min] = 0
    
    # Get staff spacing
    avg_distances = get_staff_spacing(staff_lines_groups)
    
    # Detect ellipses (notes)
    ellipse_info = detect_notes(
        img_before_detect_note,
        binary,
        repaired,
        staff_lines_groups,
        avg_distances,
        regions_stem
    )
    
    # Generate audio
    audio_path = os.path.join(output_dir, "sheet_music_output.wav")
    generate_audio(
        ellipse_info,
        audio_path,
        note_duration,
        pause_duration,
        amplitude
    )
    
    # Prepare notes information
    notes_info = []
    for e in ellipse_info:
        notes_info.append({
            "id": e.get("sorted_id", 0),
            "staff_id": e.get("staff_id", 0),
            "note_name": e.get("note_name", "Unknown"),
            "type": e.get("type", "đen"),
            "stem": e.get("stem", "không đuôi"),
            "hook": e.get("hook", None),
            "center_x": float(e.get("center_x", 0)),
            "center_y": float(e.get("center_y", 0))
        })
    
    return {
        "audio_path": audio_path,
        "note_count": len(ellipse_info),
        "staff_count": len(staff_lines_groups),
        "notes_info": notes_info
    }


def get_staff_spacing(staff_lines_groups):
    """Calculate average spacing between staff lines"""
    avg_spacings = []
    for staff in staff_lines_groups:
        staff_sorted = sorted(staff)
        distances = np.diff(staff_sorted)
        avg_spacing = np.mean(distances)
        avg_spacings.append(avg_spacing)
    return np.array(avg_spacings)


def is_black_pixel(BW, x, y, r=2, type_image=1):
    """Check if pixel is black"""
    h, w = BW.shape
    x = int(round(x))
    y = int(round(y))
    
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    
    x1 = max(0, x - r)
    x2 = min(w, x + r + 1)
    y1 = max(0, y - r)
    y2 = min(h, y + r + 1)
    
    patch = BW[y1:y2, x1:x2]
    if patch.size == 0:
        return False
    
    black_ratio = np.sum(patch == 0) / patch.size
    if type_image != 1:
        return black_ratio > 0.5
    else:
        return black_ratio < 0.5


def is_overlap(box1, box2, height):
    """Check if two boxes overlap"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    left1, right1 = x1, x1 + w1
    top1, bottom1 = y1, y1 + h1
    
    left2, right2 = x2, x2 + w2
    top2, bottom2 = y2, y2 + h2
    
    if right1 <= left2 or right2 <= left1 or bottom1 <= top2 or bottom2 <= top1:
        return False
    
    return True


def white_ratio(I_bin, box):
    """Calculate white pixel ratio in a box"""
    x, y, w, h = [int(v) for v in box]
    h_img, w_img = I_bin.shape[:2]
    
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    
    roi = I_bin[y0:y1, x0:x1]
    
    if roi.size == 0:
        return 0.0
    
    white_pixels = np.sum(roi == 255)
    total_pixels = roi.size
    
    return white_pixels / total_pixels


def detect_g_clef_boxes_for_binary(model, img_path, class_names, conf=0.05):
    """Detect G-clef using YOLO"""
    results = model.predict(img_path, conf=conf, show=False, save=False)
    
    g_clef_boxes = []
    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].cpu().numpy())
            if class_names[cls_id] == 'G-clef':
                xyxy = boxes.xyxy[i].cpu().numpy()
                g_clef_boxes.append([
                    int(xyxy[0]), int(xyxy[1]),
                    int(xyxy[2]), int(xyxy[3])
                ])
    return g_clef_boxes


def find_closest_staff(center_y, staff_groups):
    """Find the closest staff to a given y coordinate"""
    distances = [abs(center_y - np.mean(staff)) for staff in staff_groups]
    if distances:
        return np.argmin(distances)
    return None


def map_y_to_note_correct(center_y, staff_lines, staff_line_notes, noteOrder):
    """Map y coordinate to musical note"""
    staff_lines = sorted(staff_lines)
    spacing = np.mean(np.diff(staff_lines))
    yRef = staff_lines[-1]
    n_steps_from_low = round((yRef - center_y) / (spacing / 2))
    idx_base = noteOrder.index(staff_line_notes[0])
    idxNote = idx_base + n_steps_from_low
    idxNote = max(0, min(idxNote, len(noteOrder) - 1))
    return noteOrder[idxNote]


def detect_notes(img_before_detect_note, binary, repaired, staff_lines_groups, avg_distances, regions_stem):
    """Detect and classify musical notes"""
    contours, _ = cv2.findContours(img_before_detect_note, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipse_info = []
    
    noteOrder = ['A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5', 'F5', 'G5', 'A5']
    staff_line_notes = ['E4', 'F4', 'G4', 'A4', 'B4']
    
    staff_lines_groups = sorted(staff_lines_groups, key=lambda g: np.mean(g))
    
    # Collect ellipse information
    for i, cnt in enumerate(contours):
        if len(cnt) >= 5:
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)
            center_x = x_rect + w_rect / 2
            center_y = y_rect + h_rect / 2
            
            ellipse = cv2.fitEllipse(cnt)
            (x_fit, y_fit), (MA, ma), angle = ellipse
            area = cv2.contourArea(cnt)
            ratio = min(MA, ma) / max(MA, ma)
            
            if angle > 90 or angle == 0:
                continue
            if area > 120 or area < 50:
                continue
            if h_rect > 1.5 * np.mean(avg_distances):
                continue
            
            ellipse_info.append({
                'id': i,
                'center_x': center_x,
                'center_y': center_y,
                'major_axis': MA,
                'minor_axis': ma,
                'angle': angle,
                'area': area,
                'ratio': ratio,
                'w_rect': w_rect,
                'h_rect': h_rect
            })
    
    # Sort by X
    ellipse_info = sorted(ellipse_info, key=lambda e: e['center_x'])
    
    # Assign staff and note
    for e in ellipse_info:
        staff_id = find_closest_staff(e['center_y'], staff_lines_groups)
        e['staff_id'] = staff_id
        if staff_id is not None:
            e['note_name'] = map_y_to_note_correct(
                e['center_y'],
                staff_lines_groups[staff_id],
                staff_line_notes,
                noteOrder
            )
        else:
            e['note_name'] = "Unknown"
        
        e['type'] = 'đen'
        e['stem'] = 'không đuôi'
        e['hook'] = None
        
        if is_black_pixel(binary, e['center_x'], e['center_y'], 2, 1):
            e['type'] = 'đen'
        else:
            e['type'] = 'trắng'
    
    # Detect stems and hooks
    for e in ellipse_info:
        for stem in regions_stem:
            box1 = [e['center_x'], float(e['center_y'] - np.mean(avg_distances)), 
                   e['w_rect'], np.mean(avg_distances)]
            box2 = [stem['x'], stem['y'], stem['w'], stem['h']]
            box3 = [e['center_x'] - e['w_rect'], float(e['center_y'] + np.mean(avg_distances)), 
                   e['w_rect'], np.mean(avg_distances)]
            
            if is_overlap(box1, box2, np.mean(avg_distances)):
                e['stem'] = 'có đuôi'
                stem['type'] = 1
                
                x, y, w, h = stem["x"], stem["y"], stem["w"], stem["h"]
                x2 = x + w
                y2 = y
                w2 = e['w_rect']
                h2 = h
                stem_hook = white_ratio(repaired, [x2, y2, w2, h2])
                if stem_hook > 0.1:
                    e['hook'] = 'móc đơn'
                break
            
            if is_overlap(box3, box2, np.mean(avg_distances)):
                e['stem'] = 'có đuôi'
                
                x, y, w, h = stem["x"], stem["y"], stem["w"], stem["h"]
                x2 = x + w
                y2 = y + e['h_rect'] / 2
                w2 = e['w_rect']
                h2 = h
                stem_hook = white_ratio(repaired, [x2, y2, w2, h2])
                if stem_hook > 0.12:
                    e['hook'] = 'móc đơn'
                break
    
    # Sort by staff and X
    numStaff = max(e['staff_id'] for e in ellipse_info if e['staff_id'] is not None) + 1 if ellipse_info else 0
    sorted_info = []
    for s in range(numStaff):
        notes = [e for e in ellipse_info if e['staff_id'] == s]
        notes = sorted(notes, key=lambda n: n['center_x'])
        sorted_info.extend(notes)
    
    for idx, e in enumerate(sorted_info, start=1):
        e['sorted_id'] = idx
    
    return sorted_info


def generate_audio(ellipse_info, output_file, note_duration, pause_duration, amplitude):
    """Generate audio from note information"""
    fs = 44100
    fade_time = 0.01
    
    # Note frequency mapping
    noteFreqs = {
        'C4': 261.63, 'C4#': 277.18, 'D4': 293.66, 'D4#': 311.13, 'E4': 329.63, 
        'F4': 349.23, 'F4#': 369.99, 'G4': 392.00, 'G4#': 415.30, 'A4': 440.00, 
        'A4#': 466.16, 'B4': 493.88, 'C5': 523.25, 'C5#': 554.37, 'D5': 587.33, 
        'D5#': 622.25, 'E5': 659.25, 'F5': 698.46, 'F5#': 739.99, 'G5': 783.99, 
        'G5#': 830.61, 'A5': 880.00, 'B3': 246.94, 'A3': 220.00
    }
    
    fade_samples = int(fade_time * fs)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    y_sound = np.array([], dtype=np.float32)
    
    # Organize by staff
    numStaff = max(e['staff_id'] for e in ellipse_info if e['staff_id'] is not None) + 1 if ellipse_info else 0
    StaffData = []
    for s in range(numStaff):
        notes = [e for e in ellipse_info if e['staff_id'] == s]
        notes = sorted(notes, key=lambda n: n['center_x'])
        StaffData.append({'notes': notes})
    
    # Generate sound
    for s in range(numStaff):
        notes = StaffData[s]['notes']
        for n in notes:
            note_name = n['note_name']
            note_type = n.get('type', 'đen')
            hook_type = n.get('hook', None)
            stem_type = n.get('stem', 'có đuôi')
            
            f = noteFreqs.get(note_name, 440)
            
            t_dur = note_duration
            amp = amplitude
            
            if note_type == 'trắng' and stem_type == 'có đuôi':
                t_dur = note_duration * 2
                amp = amplitude * 0.7
            elif note_type == 'trắng' and stem_type == 'không đuôi':
                t_dur = note_duration * 4
                amp = amplitude * 0.6
            
            if hook_type == 'móc đơn':
                t_dur *= 0.6
            elif hook_type == 'móc kép':
                t_dur *= 0.5
            
            t = np.linspace(0, t_dur, int(fs * t_dur), endpoint=False)
            y_note = amp * np.sin(2 * np.pi * f * t)
            
            # Fade in/out
            fade_samples_note = min(fade_samples, len(y_note))
            if fade_samples_note > 0:
                y_note[:fade_samples_note] *= fade_in[:fade_samples_note]
                y_note[-fade_samples_note:] *= fade_out[:fade_samples_note]
            
            # Add pause
            silence = np.zeros(int(fs * pause_duration))
            y_sound = np.concatenate((y_sound, y_note, silence))
    
    # Normalize and save
    if len(y_sound) > 0:
        y_sound_int16 = np.int16(y_sound / np.max(np.abs(y_sound)) * 32767)
    else:
        y_sound_int16 = np.int16([])
    
    write(output_file, fs, y_sound_int16)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

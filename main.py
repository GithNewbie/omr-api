# main.py
import tempfile
import json
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from utils.image_processing import analyze_sheet
from utils.sound_generator import generate_audio

app = FastAPI(title="Sheet Music Analyzer")

OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # save upload to temp file
    suffix = os.path.splitext(file.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = analyze_sheet(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # save JSON for debug
    out_json = os.path.join(OUTPUTS_DIR, "last_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return JSONResponse(content=result)


@app.post("/audio")
async def audio_from_json(payload: dict):
    """
    Accepts the JSON produced by /analyze (or same structure) and returns a WAV file.
    """
    out_wav = os.path.join(OUTPUTS_DIR, "sheet_output.wav")
    generate_audio(payload, out_wav)
    return FileResponse(out_wav, media_type="audio/wav", filename="sheet_output.wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

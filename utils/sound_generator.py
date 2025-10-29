# utils/sound_generator.py
import numpy as np
from scipy.io.wavfile import write

NOTE_FREQS = {
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
    'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
    'C5': 523.25, 'D5': 587.33, 'E5': 659.25,
    'F5': 698.46, 'G5': 783.99, 'A5': 880.00,
    'B3': 246.94, 'A3': 220.00
}

def generate_audio(result_json: dict, output_path: str,
                   fs: int = 44100,
                   note_duration: float = 0.5,
                   pause_duration: float = 0.05,
                   amplitude: float = 0.5,
                   fade_time: float = 0.01):
    """
    Build audio from notes in result_json.
    Expects result_json["staffs"] or result_json["notes"].
    This reproduces the MATLAB ordering: iterate staff by staff,
    sorting notes in each staff by x-position.
    """
    fade_samples = int(round(fade_time * fs))
    if fade_samples < 1:
        fade_samples = 1
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    y_sound = np.array([], dtype=np.float32)

    staffs = result_json.get("staffs") or []
    if not staffs and "notes" in result_json:
        # fallback: group all into a single staff
        staffs = [{"notes": result_json["notes"]}]

    for s in staffs:
        notes = s.get("notes", [])
        # sort by x position
        notes_sorted = sorted(notes, key=lambda n: n.get("position", [0, 0])[0])
        for n in notes_sorted:
            noteName = n.get("name", "A4")
            f = NOTE_FREQS.get(noteName, 440.0)
            t = np.linspace(0, note_duration, int(round(fs * note_duration)), endpoint=False)
            y_note = amplitude * np.sin(2 * np.pi * f * t).astype(np.float32)
            # fade
            y_note[:fade_samples] *= fade_in
            y_note[-fade_samples:] *= fade_out
            silence = np.zeros(int(round(fs * pause_duration)), dtype=np.float32)
            y_sound = np.concatenate((y_sound, y_note, silence)) if y_sound.size else np.concatenate((y_note, silence))

    if y_sound.size == 0:
        # produce a tiny silence wave
        y_sound = np.zeros(int(round(fs * 0.1)), dtype=np.float32)

    # normalize
    maxv = np.max(np.abs(y_sound))
    if maxv > 0:
        y_sound = y_sound / maxv

    # convert to int16
    data = (y_sound * 32767).astype(np.int16)
    write(output_path, fs, data)

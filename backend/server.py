from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import io
import base64
import zipfile
import datetime
import uvicorn

from backend.inference import MusicEmotionPredictor, predict_multiple_files

app = FastAPI(title="ApollodB Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
predictor: Optional[MusicEmotionPredictor] = None

# Constants
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100MB
ALLOWED_MIME_PREFIX = "audio/"

# Small helpers (kept local to avoid import churn)
def _too_large(n: int) -> bool:
    return n > MAX_UPLOAD_BYTES

def _bad_mime(ct: Optional[str]) -> bool:
    return not (ct or "").startswith(ALLOWED_MIME_PREFIX)

def _write_temp(contents: bytes, filename: str) -> str:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as f:
        f.write(contents)
        return f.name

@app.on_event("startup")
async def startup_event():
    global predictor
    predictor = MusicEmotionPredictor()

@app.post("/analyze-single")
async def analyze_single(
    file: UploadFile = File(...),
    aggression: float = Form(0.5),
    eq_style: str = Form("Wavelet")
):
    # Guards
    contents = await file.read()
    if _too_large(len(contents)):
        return JSONResponse(status_code=413, content={"error": "File too large (limit 100MB)"})
    if _bad_mime(file.content_type):
        return JSONResponse(status_code=415, content={"error": "Unsupported media type. Please upload audio files."})
    # librosa requires a filename; write to temp
    temp_path = _write_temp(contents, file.filename)
    try:
        result = predictor.predict_emotion(temp_path)
        result["filename"] = file.filename
        result["temp_path"] = temp_path
        eq_data = predictor.generate_eq_curves(result["primary_emotion"], aggression)
        return {"result": result, "eq_data": eq_data, "eq_style": eq_style, "aggression": aggression}
    finally:
        # Keep temp file for potential follow-up apply-eq/spectrogram if client sends again
        pass

@app.post("/analyze-batch")
async def analyze_batch(
    files: List[UploadFile] = File(...),
    aggression: float = Form(0.5),
    eq_style: str = Form("Wavelet")
):
    temp_paths = []
    filenames = []
    try:
        for f in files:
            data = await f.read()
            if _too_large(len(data)):
                return JSONResponse(status_code=413, content={"error": f"File too large: {f.filename}"})
            if _bad_mime(f.content_type):
                return JSONResponse(status_code=415, content={"error": f"Unsupported media type: {f.filename}"})
            temp_paths.append(_write_temp(data, f.filename))
            filenames.append(f.filename)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    results = predict_multiple_files(temp_paths, predictor)
    # attach filenames and temp paths to individual results
    for i, r in enumerate(results.get("individual_results", [])):
        r["filename"] = filenames[i]
        r["temp_path"] = temp_paths[i]
    eq_data = predictor.generate_eq_curves(results["dominant_emotion"], aggression)
    return {"results": results, "eq_data": eq_data, "eq_style": eq_style, "aggression": aggression}

@app.post("/apply-eq")
async def apply_eq(
    file: UploadFile = File(...),
    emotion: str = Form(...),
    aggression: float = Form(0.5),
    confidence: float = Form(None),
    warmth: float = Form(0.0),
    presence: float = Form(0.0),
    air: float = Form(0.0),
    hq_linear: str = Form("false"),
):
    data = await file.read()
    if _too_large(len(data)):
        return JSONResponse(status_code=413, content={"error": "File too large (limit 100MB)"})
    if _bad_mime(file.content_type):
        return JSONResponse(status_code=415, content={"error": "Unsupported media type. Please upload audio files."})
    path = _write_temp(data, file.filename)
    # Robust boolean parsing for form fields
    def parse_bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        return str(v).strip().lower() in {"1","true","yes","on"}

    result = predictor.apply_eq_to_audio(path, emotion, aggression,
                                         confidence=confidence,
                                         warmth=warmth, presence=presence, air=air,
                                         hq_linear=parse_bool(hq_linear))
    if not result:
        return JSONResponse(status_code=500, content={"error": "Failed to process audio"})
    if isinstance(result, tuple):
        audio_bytes, stats = result
    else:
        audio_bytes, stats = result, {}
    headers = {
        "Content-Disposition": f"attachment; filename=eqed_{emotion}_{file.filename.split('.')[0]}.wav",
        "Cache-Control": "no-store, no-cache, must-revalidate, private",
    }
    if stats:
        tp = stats.get("true_peak_dbtp")
        lufs = stats.get("lufs")
        if tp is not None:
            headers["X-True-Peak-DBTP"] = str(tp)
        if lufs is not None:
            headers["X-LUFS"] = str(lufs)
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav", headers=headers)

@app.post("/spectrogram")
async def spectrogram(file: UploadFile = File(...)):
    data = await file.read()
    if _too_large(len(data)):
        return JSONResponse(status_code=413, content={"error": "File too large (limit 100MB)"})
    if _bad_mime(file.content_type):
        return JSONResponse(status_code=415, content={"error": "Unsupported media type. Please upload audio files."})
    import os
    path = _write_temp(data, file.filename)
    img_path = predictor.generate_spectrogram(path)
    if not img_path:
        return JSONResponse(status_code=500, content={"error": "Failed to generate spectrogram"})
    with open(img_path, "rb") as imgf:
        b64 = base64.b64encode(imgf.read()).decode("ascii")
    # cleanup image file best-effort
    try:
        os.unlink(img_path)
    except Exception:
        pass
    return {"image_base64": f"data:image/png;base64,{b64}"}

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "version": app.version}


@app.post("/apply-eq-batch-zip")
async def apply_eq_batch_zip(
    files: List[UploadFile] = File(...),
    emotions: List[str] = Form(...),
    confidences: List[str] = Form(None),
    aggression: float = Form(0.5),
    warmth: float = Form(0.0),
    presence: float = Form(0.0),
    air: float = Form(0.0),
    hq_linear: str = Form("false"),
):
    # Validate lengths
    if len(files) != len(emotions):
        return JSONResponse(status_code=400, content={"error": "files and emotions length mismatch"})
    # Boolean parse
    def parse_bool(v):
        return str(v).strip().lower() in {"1","true","yes","on"}
    hq = parse_bool(hq_linear)
    # Prepare ZIP in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # optional provenance
        timestamp = datetime.datetime.utcnow().isoformat() + 'Z'
        manifest = [f"ApollodB Batch EQ {timestamp}", f"Aggression: {aggression}", f"HQ Linear: {hq}", f"Warmth: {warmth}", f"Presence: {presence}", f"Air: {air}"]
        zf.writestr('MANIFEST.txt', "\n".join(manifest))
        for i, f in enumerate(files):
            data = await f.read()
            if _too_large(len(data)):
                return JSONResponse(status_code=413, content={"error": f"File too large: {f.filename}"})
            if _bad_mime(f.content_type):
                return JSONResponse(status_code=415, content={"error": f"Unsupported media type: {f.filename}"})
            # write temp file
            path = _write_temp(data, f.filename)
            conf = None
            try:
                if confidences is not None:
                    conf_raw = confidences[i] if i < len(confidences) else None
                    if conf_raw is not None and str(conf_raw).strip() != '':
                        conf = float(conf_raw)
            except Exception:
                conf = None
            result = predictor.apply_eq_to_audio(path, emotions[i], aggression,
                                                 confidence=conf,
                                                 warmth=warmth, presence=presence, air=air,
                                                 hq_linear=hq)
            if not result:
                return JSONResponse(status_code=500, content={"error": f"Failed to process: {f.filename}"})
            if isinstance(result, tuple):
                audio_bytes, stats = result
            else:
                audio_bytes, stats = result, {}
            out_name = f"eqed_{emotions[i]}_{(f.filename or f'file_{i+1}').split('.')[0]}.wav"
            zf.writestr(out_name, audio_bytes)
            # sidecar with stats
            if stats:
                side = []
                for k, v in stats.items():
                    side.append(f"{k}: {v}")
                zf.writestr(out_name.replace('.wav', '_stats.txt'), "\n".join(side))
    buf.seek(0)
    headers = {
        "Content-Disposition": "attachment; filename=apollodb_batch_eq.zip",
        "Cache-Control": "no-store, no-cache, must-revalidate, private",
    }
    return StreamingResponse(buf, media_type='application/zip', headers=headers)


@app.post("/export-eq")
async def export_eq(
    emotion: str = Form(...),
    aggression: float = Form(0.5),
    fmt: str = Form("apo"),  # apo | autoeq_json | autoeq_csv
):
    eq = predictor.generate_eq_curves(emotion, aggression)
    wavelet = eq.get('wavelet', '')
    param = eq.get('parametric', '')
    fmt = (fmt or 'apo').lower()
    if fmt == 'apo':
        # Equalizer APO accepts GraphicEQ lines, include parametric as comments
        content = [wavelet]
        if param and param != 'Flat':
            content.append('')
            content.append('; Parametric summary:')
            for line in param.splitlines():
                content.append(f"; {line}")
        data = "\n".join(content).encode('utf-8')
        return StreamingResponse(io.BytesIO(data), media_type='text/plain', headers={
            'Content-Disposition': f'attachment; filename=eq_{emotion}_apo.txt',
            'Cache-Control': 'no-store'
        })
    elif fmt == 'autoeq_json':
        # Emit frequency->gain pairs as JSON
        import json
        s = wavelet.replace('GraphicEQ:', '').strip()
        pairs = []
        for seg in filter(None, [p.strip() for p in s.split(';')]):
            try:
                f, g = seg.split()
                pairs.append({"f": float(f), "g_db": float(g)})
            except Exception:
                pass
        data = json.dumps({"emotion": emotion, "aggression": aggression, "points": pairs}).encode('utf-8')
        return StreamingResponse(io.BytesIO(data), media_type='application/json', headers={
            'Content-Disposition': f'attachment; filename=eq_{emotion}_autoeq.json',
            'Cache-Control': 'no-store'
        })
    elif fmt == 'autoeq_csv':
        # Emit CSV: frequency,g_db
        import csv
        bio = io.StringIO()
        w = csv.writer(bio)
        w.writerow(['frequency','g_db'])
        s = wavelet.replace('GraphicEQ:', '').strip()
        for seg in filter(None, [p.strip() for p in s.split(';')]):
            try:
                f, g = seg.split()
                w.writerow([float(f), float(g)])
            except Exception:
                pass
        data = bio.getvalue().encode('utf-8')
        return StreamingResponse(io.BytesIO(data), media_type='text/csv', headers={
            'Content-Disposition': f'attachment; filename=eq_{emotion}_autoeq.csv',
            'Cache-Control': 'no-store'
        })
    else:
        return JSONResponse(status_code=400, content={"error": "Unknown format"})

if __name__ == "__main__":
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)

# Mount static site for convenience and Cloud Run demo at /app
app.mount("/app", StaticFiles(directory="web", html=True), name="app")

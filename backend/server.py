import os
import io 
import base64
import zipfile
import datetime
import uvicorn
import logging
import tempfile
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Import the new classifier
from backend.inference_classifier import MusicEmotionPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ApollodB Backend", version="1.0.0")

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to the client
)

# Add middleware to log incoming requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Load model once
predictor: Optional[MusicEmotionPredictor] = None

# Model configuration
MODEL_CONFIG = {
    'model_path': 'models/best_val_loss.pth',
    'labels_path': 'models/labels.json',
    'scaler_mean_path': 'models/scaler_mean.npy',
    'scaler_scale_path': 'models/scaler_scale.npy'
}

# Constants
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100MB
ALLOWED_MIME_PREFIX = "audio/"

# Small helpers (kept local to avoid import churn)
def _too_large(n: int) -> bool:
    return n > MAX_UPLOAD_BYTES

def _bad_mime(ct: Optional[str]) -> bool:
    # Accept audio/* and also application/octet-stream (common for curl uploads)
    if not ct:
        return False
    if ct.startswith(ALLOWED_MIME_PREFIX):
        return False
    if ct == "application/octet-stream":
        return False
    return True

def _write_temp(contents: bytes, filename: str) -> str:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as f:
        f.write(contents)
        return f.name

def get_predictor():
    """Lazy load the predictor to save memory."""
    global predictor
    if predictor is None:
        try:
            predictor = MusicEmotionPredictor(
                model_path=MODEL_CONFIG['model_path'],
                labels_path=MODEL_CONFIG['labels_path'],
                scaler_mean_path=MODEL_CONFIG['scaler_mean_path'],
                scaler_scale_path=MODEL_CONFIG['scaler_scale_path']
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            predictor = None
            raise
    return predictor

@app.on_event("startup")
async def startup_event():
    """Initialize the predictor on startup."""
    try:
        # Just verify we can load the predictor, but don't keep it in memory
        pred = get_predictor()
        # Clean up immediately to save memory
        pred.cleanup()
        print("Model verified successfully")
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise

@app.post("/analyze-single")
async def analyze_single(
    request: Request,
    file: UploadFile = File(...),
    aggression: float = Form(0.5),
    eq_style: str = Form("Wavelet")
):
    predictor = None
    temp_file = None
    
    try:
        # Get predictor instance
        predictor = get_predictor()
        
        # Check file size and type
        file_contents = await file.read()
        if _too_large(len(file_contents)):
            raise HTTPException(status_code=413, detail=f"File too large. Max size is {MAX_UPLOAD_BYTES} bytes")
        
        if _bad_mime(file.content_type):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")
        
        # Save to temp file
        temp_file = _write_temp(file_contents, file.filename or "audio")
        
        # Get prediction
        result = predictor.predict_emotion(temp_file)
        
        # Generate EQ curves
        eq = predictor.generate_eq_curves(
            result.get("primary_emotion") or result.get("emotion"),
            aggression,
            valence=result.get("valence"),
            arousal=result.get("arousal"),
            confidence=result.get("confidence"),
            secondary=result.get("secondary_emotion"),
        )
        
        response = {
            "emotion": {
                "primary_emotion": result.get("primary_emotion") or result.get("emotion"),
                "secondary_emotion": result.get("secondary_emotion"),
                "valence": float(result.get("valence", 0.5)),
                "arousal": float(result.get("arousal", 0.5)),
                "confidence": float(result.get("confidence", 0.0)),
                "probabilities": result.get("probabilities", {}),
            },
            "eq_data": eq,
            "eq_style": eq_style,
            "aggression": aggression,
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in analyze_single: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {str(e)}")
        
        # Clean up predictor to free memory
        if predictor:
            try:
                predictor.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up predictor: {str(e)}")
        try:
            os.unlink(tmp_path)
        except Exception as e:
            print(f"Error cleaning up temp file: {e}")

@app.post("/api/analyze-single")
async def analyze_single_api(
    request: Request,
    aggression: float = Form(0.5),
    eq_style: str = Form("Wavelet")
):
    # Flexible handler: accept any of these field names, or the first UploadFile in the form
    form = await request.form()
    upload = None
    def is_upload(x):
        # Accept both FastAPI/Starlette UploadFile or any object with filename+read
        return hasattr(x, "filename") and hasattr(x, "read")
    # Try common field names
    for key in ("file", "audio", "track", "upload"):
        if key in form:
            v = form[key]
            if is_upload(v):
                upload = v
                break
            # Some frameworks send lists
            if isinstance(v, list):
                for item in v:
                    if is_upload(item):
                        upload = item
                        break
                if upload is not None:
                    break
    # Fallback: first UploadFile-like value anywhere in form
    if upload is None:
        for v in form.values():
            if is_upload(v):
                upload = v
                break
    if upload is None:
        return JSONResponse(status_code=400, content={"error": "No audio file uploaded. Expected field 'file' or 'audio'."})
    contents = await upload.read()
    if _too_large(len(contents)):
        return JSONResponse(status_code=413, content={"error": "File too large (limit 100MB)"})
    # MIME check is lenient; already relaxed in _bad_mime
    if _bad_mime(upload.content_type):
        # Try to continue even if unknown content-type
        pass
    temp_path = _write_temp(contents, upload.filename or "upload.wav")
    try:
        print(f"[analyze-single] start filename={upload.filename} size={len(contents)}")
        result = predictor.predict_emotion(temp_path)
        result["filename"] = upload.filename
        result["temp_path"] = temp_path
        eq_data = predictor.generate_eq_curves(
            result["primary_emotion"],
            aggression,
            valence=result.get("valence"),
            arousal=result.get("arousal"),
            confidence=result.get("confidence"),
            secondary=result.get("secondary_emotion"),
        )
        print(f"[analyze-single] done filename={upload.filename} emotion={result['primary_emotion']}")
        return {"result": result, "eq_data": eq_data, "eq_style": eq_style, "aggression": aggression}
    finally:
        pass

# Aliases for common frontend expectations
@app.post("/analyze")
async def analyze_alias(request: Request, aggression: float = Form(0.5), eq_style: str = Form("Wavelet")):
    return await analyze_single_api(request, aggression, eq_style)

@app.post("/api/analyze")
async def analyze_api_alias(request: Request, aggression: float = Form(0.5), eq_style: str = Form("Wavelet")):
    return await analyze_single_api(request, aggression, eq_style)

@app.post("/analyze-batch")
async def analyze_batch(
    files: List[UploadFile] = File(...),
    aggression: float = Form(0.5),
    eq_style: str = Form("Wavelet")
):
    # Collect individual results
    individuals = []
    for file in files:
        data = await file.read()
        if _too_large(len(data)):
            individuals.append({"filename": file.filename, "error": "File too large"})
            continue
        # Allow unknown content-type; many browsers omit for dragged files
        path = _write_temp(data, file.filename or "upload.wav")
        try:
            r = predictor.predict_emotion(path)
            # Ensure expected keys
            indiv = {
                "filename": file.filename,
                "primary_emotion": r.get("primary_emotion") or r.get("emotion"),
                "secondary_emotion": r.get("secondary_emotion"),
                "valence": float(r.get("valence", 0.5)),
                "arousal": float(r.get("arousal", 0.5)),
                "confidence": float(r.get("confidence", 0.0)),
                "probabilities": r.get("probabilities", {}),
            }
            individuals.append(indiv)
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            individuals.append({"filename": file.filename, "error": str(e)})
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    # Aggregate statistics
    valid = [x for x in individuals if not x.get("error")]
    total = len(individuals)
    if valid:
        avg_valence = sum(x["valence"] for x in valid) / max(len(valid), 1)
        avg_arousal = sum(x["arousal"] for x in valid) / max(len(valid), 1)
        # Emotion distribution by primary
        dist: Dict[str, int] = {}
        for x in valid:
            emo = (x["primary_emotion"] or "unknown").lower()
            dist[emo] = dist.get(emo, 0) + 1
        dominant = max(dist.items(), key=lambda kv: kv[1])[0] if dist else "neutral"
    else:
        avg_valence = 0.5
        avg_arousal = 0.5
        dist = {}
        dominant = "neutral"

    # Aggregate EQ using average VA to match original logic
    eq_data = predictor.generate_eq_curves(
        dominant,
        aggression,
        valence=avg_valence,
        arousal=avg_arousal,
        confidence=None,
        secondary=None,
    )

    payload = {
        "results": {
            "total_songs": total,
            "dominant_emotion": dominant,
            "average_valence": avg_valence,
            "average_arousal": avg_arousal,
            "emotion_distribution": dist,
            "individual_results": valid,
        },
        "eq_data": eq_data,
        "eq_style": eq_style,
        "aggression": aggression,
    }
    return payload

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
    logger.info("Health check endpoint called")
    return {"status": "ok", "version": app.version}


@app.post("/export-eq")
async def export_eq(
    emotion: str = Form(...),
    aggression: float = Form(0.5),
    fmt: str = Form("apo"),  # apo | autoeq_json | autoeq_csv
    valence: float = Form(None),
    arousal: float = Form(None),
    confidence: float = Form(None),
    secondary: str = Form(None),
):
    # Generate EQ preset from emotion
    eq = predictor.generate_eq_curves(
        emotion,
        aggression,
        valence=valence,
        arousal=arousal,
        confidence=confidence,
        secondary=secondary,
    )
    wavelet = eq.get("wavelet", "GraphicEQ: 20 0; 20000 0")
    # Convert wavelet string to different formats
    def parse_wavelet(s: str):
        s = s.replace("GraphicEQ:", "").strip()
        pairs = [p.strip() for p in s.split(";") if p.strip()]
        out = []
        for p in pairs:
            parts = p.split()
            if len(parts) >= 2:
                try:
                    out.append((float(parts[0]), float(parts[1])))
                except Exception:
                    pass
        return out
    pts = parse_wavelet(wavelet)
    if fmt == "apo":
        # Equalizer APO GraphicEQ format passthrough
        text = wavelet if wavelet.startswith("GraphicEQ:") else f"GraphicEQ: " + "; ".join(f"{int(f)} {g:.2f}" for f,g in pts)
        return StreamingResponse(io.BytesIO(text.encode("utf-8")), media_type="text/plain", headers={
            "Content-Disposition": f"attachment; filename=eq_{emotion}_apo.txt"
        })
    elif fmt == "autoeq_json":
        import json as _json
        obj = {"frequency": [f for f,_ in pts], "gain": [g for _,g in pts], "unit": "dB"}
        data = _json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return StreamingResponse(io.BytesIO(data), media_type="application/json", headers={
            "Content-Disposition": f"attachment; filename=eq_{emotion}_autoeq.json"
        })
    elif fmt == "autoeq_csv":
        import csv
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["frequency", "gain_db"]) 
        for f, g in pts:
            w.writerow([f, g])
        data = buf.getvalue().encode("utf-8")
        return StreamingResponse(io.BytesIO(data), media_type="text/csv", headers={
            "Content-Disposition": f"attachment; filename=eq_{emotion}_autoeq.csv"
        })
    else:
        return JSONResponse(status_code=400, content={"error": "Unknown format"})

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


# Serve index.html at root
@app.get("/")
async def root_index():
    return FileResponse("web/index.html")

# Serve index.html at /index.html
@app.get("/index.html")
async def index_html():
    return FileResponse("web/index.html")

# Serve static files from /static
app.mount("/static", StaticFiles(directory="web"), name="static")

# Add a catch-all route to serve the frontend app
@app.get("/{full_path:path}")
async def catch_all(request: Request, full_path: str):
    # Try to serve the file if it exists
    file_path = Path("web") / full_path
    if file_path.is_file():
        return FileResponse(file_path)
    # Otherwise serve index.html for SPA routing
    return FileResponse("web/index.html")


# Ensure all routes are defined before this point

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("backend.server:app", host="0.0.0.0", port=port, reload=False)

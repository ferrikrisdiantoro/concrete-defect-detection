from __future__ import annotations

import io, os, json, logging, uuid, threading
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

from storage import (
    uploads_dir,
    ensure_storage_ready,
    load_history,
    append_history,
    update_history_record,
    delete_history_record,
    delete_history_filtered,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("crack-detector")

# -----------------------------------------------------------------------------
# KONFIG TANPA ENV (hardcoded)
# -----------------------------------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "crack-detection-model.onnx")
# Path relative to src/backend/app.py -> root -> models/production
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "production", "crack-detection-model.onnx")

CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.45
SAVE_UPLOADS = 1
SAVE_HISTORY = 1
ALLOW_ORIGINS = ["*"]    # CORS

# Nama kelas (URUTAN harus sesuai model)
CLASS_NAMES: Dict[int, str] = {
    0: "crack",
    1: "spalling",
    2: "honeycomb",
    3: "segregation",
    4: "corrosion",
}

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
ensure_storage_ready()

# -----------------------------------------------------------------------------
# Severity Rules & Solutions
# -----------------------------------------------------------------------------
SUPPORTED_TYPES = {"corrosion", "crack", "honeycomb", "segregation", "spalling"}

DEFAULT_THRESHOLDS = {
    "crack":       {"ringan": 0.0000, "sedang": 0.0050, "berat": 0.0200},
    "honeycomb":   {"ringan": 0.0000, "sedang": 0.0040, "berat": 0.0150},
    "spalling":    {"ringan": 0.0000, "sedang": 0.0030, "berat": 0.0120},
    "corrosion":   {"ringan": 0.0000, "sedang": 0.0040, "berat": 0.0100},
    "segregation": {"ringan": 0.0000, "sedang": 0.0060, "berat": 0.0180},
}

SOLUTION_RULES: Dict[Tuple[str, str], str] = {
    ("corrosion", "ringan"): "Clean surface (wire brushing) + protective coating (sealant/inhibitor).",
    ("corrosion", "sedang"): "Patch vulnerable areas (mortar repair) + antiseptic protection & carbonation protective layer.",
    ("corrosion", "berat"):  "Structural repair: grouting/patching, reinforcement (jacket/FRP), + anti-corrosion coating.",
    ("crack", "ringan"): "Surface grinding + thin V-cut + skim coat (superficial repair).",
    ("crack", "sedang"): "Epoxy injection; V-cut along crack, then inject epoxy.",
    ("crack", "berat"):  "Epoxy injection + external reinforcement (CFRP/plating) or reconstruction if necessary.",
    ("honeycomb", "ringan"): "Patching mortar for thin & superficial porous areas.",
    ("honeycomb", "sedang"): "Grouting (injection/pour mortar) + surface preparation & curing.",
    ("honeycomb", "berat"):  "Grouting/major repair (larger coverage/volume).",
    ("segregation", "ringan"): "Patch/skim patch to even out appearance & homogeneity.",
    ("segregation", "sedang"): "Local grouting to fill less dense areas.",
    ("segregation", "berat"):  "Re-casting of highly non-homogeneous areas / mix correction.",
    ("spalling", "ringan"): "Patch small areas; clean surface & free from oil.",
    ("spalling", "sedang"): "Chipping + patching; mortar layer 1–2 cm; rebar protection.",
    ("spalling", "berat"):  "Demolish damaged concrete + re-pour; ensure bonding & corrosion protection.",
}

GENERAL_VISUAL_CUES: Dict[str, List[str]] = {
    "corrosion":   ["Rust color", "Defects around rebar", "Peeling/flaking texture"],
    "crack":       ["Measure crack width from bbox/mask", "Observe crack texture & contrast"],
    "honeycomb":   ["Porous area + visible aggregate", "Rough/uneven surface", "Dark/deep areas"],
    "segregation": ["Color/texture gradation", "Vertical layer separation", "Non-homogeneous surface"],
    "spalling":    ["Indentation/hole defects", "Exposed reinforcement", "Cracked & brittle surface"],
}

SEVERITY_FEATURES: Dict[Tuple[str, str], List[str]] = {
    ("corrosion", "ringan"): ["Brown/orange discoloration", "No cracking/spalling", "Rebar not visible"],
    ("corrosion", "sedang"): ["Even rust + flow stains", "Fine cracks (≤0.5 mm) around rebar", "No spalling yet"],
    ("corrosion", "berat"):  ["Exposed rebar", "Heavy spalling/corrosion", "Large/longitudinal cracks"],
    ("crack", "ringan"): ["Width ≤0.3 mm (hairline)", "Linear, not branching"],
    ("crack", "sedang"): ["Width 0.3–1 mm", "Starts branching/spreading"],
    ("crack", "berat"):  ["Width >1 mm", "Crossing/spreading widely"],
    ("honeycomb", "ringan"): ["Small cavities <10 mm, local", "Not deep, rebar not visible"],
    ("honeycomb", "sedang"): ["Cavities 10–30 mm, multiple", "Approaching rebar"],
    ("honeycomb", "berat"):  ["Cavities >30 mm, deep", "Large aggregate visible/rebar near surface"],
    ("segregation", "ringan"): ["Slightly uneven texture", "Cement paste dominant"],
    ("segregation", "sedang"): ["Aggregate down, paste up", "Surface not homogeneous"],
    ("segregation", "berat"):  ["Clear layer separation", "Potential cracks/local weakness"],
    ("spalling", "ringan"): ["Flaking ≤10 mm (small area)", "Not yet penetrating rebar"],
    ("spalling", "sedang"): ["Thickness 10–30 mm, large area", "Approaching/exposing rebar"],
    ("spalling", "berat"):  ["Flaking >30 mm (deep)", "Rebar exposed"],
}

SEV_ID_TO_EN = {"ringan": "Minor", "sedang": "Moderate", "berat": "Severe"}
SEV_EN_TO_ID = {"minor": "ringan", "moderate": "sedang", "severe": "berat"}

# -----------------------------------------------------------------------------
# Utils Severity
# -----------------------------------------------------------------------------
def parse_type_and_severity(raw_name: str) -> Tuple[str, Optional[str]]:
  name = raw_name.lower().strip()
  # Handle Indonesian keywords just in case
  name = name.replace("korosi", "corrosion").replace("retak", "crack")
  parts = name.split("_")
  if len(parts) >= 2:
      sev = parts[1]
      if sev in {"ringan","sedang","berat"}:
          sv = sev
      else:
          sv = SEV_EN_TO_ID.get(sev, None)
      if parts[0] in SUPPORTED_TYPES and sv in {"ringan","sedang","berat"}:
          return parts[0], sv
  for t in SUPPORTED_TYPES:
      if name.startswith(t) or t in name:
          return t, None
  return name, None

def infer_severity_by_ratio(dmg_type: str, area_ratio: float) -> str:
    th = DEFAULT_THRESHOLDS.get(dmg_type, DEFAULT_THRESHOLDS["crack"])
    if area_ratio >= th["berat"]: return "berat"
    if area_ratio >= th["sedang"]: return "sedang"
    return "ringan"

def solution_for(dmg_type: str, severity_id: str) -> str:
    return SOLUTION_RULES.get((dmg_type, severity_id), "Further inspection & structural consultation required.")

def build_keterangan(dmg_type: str, severity_id: str) -> str:
    cues = GENERAL_VISUAL_CUES.get(dmg_type, [])
    sev_feats = SEVERITY_FEATURES.get((dmg_type, severity_id), [])
    lines: List[str] = []
    if sev_feats:
        lines.append("Indicators:"); lines.extend([f"• {s}" for s in sev_feats])
    if cues:
        if lines: lines.append("")
        lines.append("Visual Cues:"); lines.extend([f"• {s}" for s in cues])
    return "\n".join(lines) if lines else ""

def to_severity_en(sev_id_or_en: str) -> str:
    s = sev_id_or_en.lower().strip()
    if s in SEV_ID_TO_EN: return SEV_ID_TO_EN[s]
    return s.capitalize()

# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class Dimensions(BaseModel): width: float; depth: float
class Coordinates(BaseModel): x: float; y: float; z: float

class ColumnMeta(BaseModel):
    project_id: str
    project_name: Optional[str] = None
    column_id: str
    level: Optional[str] = None
    grid: Optional[str] = None
    location: Optional[str] = None
    dimensions_mm: Optional[Dimensions] = None
    material_spec: Optional[str] = None
    year_of_construction: Optional[int] = None
    defect_side: Optional[Literal["North","South","East","West"]] = None
    inspection_date: Optional[date] = None
    inspector: Optional[str] = None
    column_type: Optional[str] = None
    notes: Optional[str] = None
    coordinates_mm: Optional[Coordinates] = None

class Monitoring(BaseModel):
    status: Literal["pending","in_progress","fixed"] = "pending"
    notes: Optional[str] = ""
    photos: List[str] = Field(default_factory=list)
    updated_at: Optional[str] = None

class DetectionItem(BaseModel):
    bbox: List[int] = Field(..., description="[x1,y1,x2,y2] px")
    conf: float
    cls: str
    severity: str
    solution: str
    keterangan: str
    project_id: str
    column_id: str
    level: Optional[str] = ""
    grid: Optional[str] = ""
    location: Optional[str] = ""
    timestamp: str
    monitoring: Optional[Monitoring] = None

class AnalyzeResponse(BaseModel):
    image_width: int
    image_height: int
    items: List[DetectionItem]

class ExportPayload(BaseModel):
    project_id: str
    items: List[DetectionItem]

# -----------------------------------------------------------------------------
# FastAPI App & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="Crack Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Static uploads
app.mount("/uploads", StaticFiles(directory=uploads_dir()), name="uploads")

# -----------------------------------------------------------------------------
# ONNX Runtime Loader & Inference (ringkas)
# -----------------------------------------------------------------------------
LAST_ONNX_ERROR = ""
OS_HAS_ORT = False
ORT_SESSION = None
ORT_INPUT_NAME = None
_MODEL_LOCK = threading.Lock()

def _load_onnx_session():
    global OS_HAS_ORT, ORT_SESSION, ORT_INPUT_NAME, LAST_ONNX_ERROR
    if ORT_SESSION is not None:
        return ORT_SESSION
    with _MODEL_LOCK:
        if ORT_SESSION is not None:
            return ORT_SESSION
        try:
            import onnxruntime as ort
            OS_HAS_ORT = True
        except Exception as e:
            LAST_ONNX_ERROR = f"import onnxruntime failed: {e}"
            OS_HAS_ORT = False
            log.warning("onnxruntime tidak tersedia; pakai STUB. (%s)", e)
            return None

        if not os.path.exists(MODEL_PATH):
            LAST_ONNX_ERROR = f"model not found at {MODEL_PATH}"
            log.warning(LAST_ONNX_ERROR)
            return None
        try:
            sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
            ORT_SESSION = sess
            ORT_INPUT_NAME = sess.get_inputs()[0].name
            LAST_ONNX_ERROR = ""
            log.info("ONNX model loaded: %s | input name: %s", MODEL_PATH, ORT_INPUT_NAME)
            return ORT_SESSION
        except Exception as e:
            LAST_ONNX_ERROR = f"InferenceSession failed: {e}"
            log.exception("Gagal load ONNX: %s", e)
            ORT_SESSION = None
            return None

def _letterbox(im: Image.Image, new_size=640, color=(114,114,114)):
    w0, h0 = im.size
    r = min(new_size / w0, new_size / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    im_resized = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (new_size, new_size), color)
    padw = (new_size - nw) // 2
    padh = (new_size - nh) // 2
    canvas.paste(im_resized, (padw, padh))
    return canvas, r, padw, padh

def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres=0.45):
    if len(boxes) == 0: return []
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

def _to_img_coords_xyxy(boxes: np.ndarray, r: float, padw: int, padh: int, w0: int, h0: int) -> np.ndarray:
    boxes = boxes.astype(np.float32).copy()
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - padw) / r).clip(0, w0 - 1)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - padh) / r).clip(0, h0 - 1)
    return boxes

def run_inference(image_bytes: bytes):
    img0 = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w0, h0 = img0.size
    im, r, padw, padh = _letterbox(img0, 640)

    arr = np.asarray(im).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]

    sess = _load_onnx_session()
    if sess is None:
        bw, bh = int(0.25 * w0), int(0.25 * h0)
        x1 = int((w0 - bw) / 2); y1 = int((h0 - bh) / 2)
        return w0, h0, [{"bbox":[x1,y1,x1+bw,y1+bh], "conf":0.88, "cls_name":"spalling"}], None

    try:
        out_any = sess.run(None, {ORT_INPUT_NAME: arr})
    except Exception as e:
        global LAST_ONNX_ERROR
        LAST_ONNX_ERROR = f"infer failed: {e}"
        log.exception("ONNX infer error")
        bw, bh = int(0.25 * w0), int(0.25 * h0)
        x1 = int((w0 - bw) / 2); y1 = int((h0 - bh) / 2)
        return w0, h0, [{"bbox":[x1,y1,x1+bw,y1+bh], "conf":0.88, "cls_name":"spalling"}], None

    out = out_any[0]
    if out.ndim == 3:
        preds = out[0] if out.shape[1] > out.shape[2] else out[0].T
    elif out.ndim == 2:
        preds = out
    else:
        preds = out.reshape(-1, out.shape[-1])

    log.info(f"Model Output Shape: {preds.shape}")
    dets: List[Dict[str, Any]] = []
    try:
        C = preds.shape[1]
        L = 640.0
        if C == 6:
            boxes = preds[:, :4].astype(np.float32)
            scores = preds[:, 4].astype(np.float32)
            cls_ids = preds[:, 5].astype(int)
            boxes = boxes * L
            if len(scores) > 0:
                log.info(f"Max score (format 1): {scores.max()}")
            m = scores >= CONF_THRESHOLD
            boxes, scores, cls_ids = boxes[m], scores[m], cls_ids[m]
            boxes = _to_img_coords_xyxy(boxes, r, padw, padh, w0, h0)
            keep = _nms_xyxy(boxes, scores, IOU_THRESHOLD)
            for i in keep:
                dets.append({
                    "bbox": boxes[i].astype(int).tolist(),
                    "conf": float(scores[i]),
                    "cls_name": CLASS_NAMES.get(int(cls_ids[i]), f"class_{int(cls_ids[i])}")
                })
        elif C >= 6:
            b = preds[:, :4].astype(np.float32)
            obj = preds[:, 4].astype(np.float32)
            cls_scores = preds[:, 5:] if C > 6 else np.ones((preds.shape[0], 1), dtype=np.float32)
            cls_ids = cls_scores.argmax(1)
            cls_conf = cls_scores.max(1)
            scores = obj * cls_conf
            if len(scores) > 0:
                log.info(f"Max score (format 2): {scores.max()}")
            scale = L if b.max() <= 1.0001 else 1.0
            boxes_xywh = np.stack([
                b[:, 0] - b[:, 2] / 2.0,
                b[:, 1] - b[:, 3] / 2.0,
                b[:, 0] + b[:, 2] / 2.0,
                b[:, 1] + b[:, 3] / 2.0
            ], 1) * scale
            boxes_xyxy = (b * scale)
            boxes_xywh = _to_img_coords_xyxy(boxes_xywh, r, padw, padh, w0, h0)
            boxes_xyxy = _to_img_coords_xyxy(boxes_xyxy, r, padw, padh, w0, h0)
            v1 = (boxes_xywh[:, 2] > boxes_xywh[:, 0]) & (boxes_xywh[:, 3] > boxes_xywh[:, 1])
            v2 = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
            boxes = boxes_xyxy if v2.sum() >= v1.sum() else boxes_xywh
            m = scores >= CONF_THRESHOLD
            boxes, scores, cls_ids = boxes[m], scores[m], cls_ids[m]
            keep = _nms_xyxy(boxes, scores, IOU_THRESHOLD)
            for i in keep:
                dets.append({
                    "bbox": boxes[i].astype(int).tolist(),
                    "conf": float(scores[i]),
                    "cls_name": CLASS_NAMES.get(int(cls_ids[i]), f"class_{int(cls_ids[i])}")
                })
    except Exception as e:
        log.exception("ONNX postprocess error: %s", e)

    # --- DEBUG LOGGING: Print top 10 raw predictions before NMS/Threshold ---
    try:
        if 'scores' in locals() and 'cls_ids' in locals():
            # Gabungkan scores dan cls_ids untuk sorting
            # Note: scores disini mungkin sudah difilter threshold di blok kode atas,
            # jadi kita coba akses raw data jika memungkinkan, atau print yang lolos threshold minimal.
            # Tapi blok atas sudah overwrite variabel 'scores'.
            # Untuk debug maksimal, kita print apa yang ada di 'dets' (yang lolos NMS)
            # DAN kita print statistik raw jika memungkinkan.
            pass

        if len(dets) == 0:
            log.info("check_debug: Tidak ada deteksi yang lolos threshold & NMS.")
        else:
            log.info(f"check_debug: {len(dets)} deteksi lolos (final):")
            for d in dets:
                log.info(f" > {d['cls_name']} : {d['conf']:.4f} | box: {d['bbox']}")

    except Exception as e:
        log.error(f"Error printing debug: {e}")
    # ------------------------------------------------------------------------

    return w0, h0, dets, None

# -----------------------------------------------------------------------------
# Health / Diag / LB Check
# -----------------------------------------------------------------------------
@app.get("/_lbcheck", response_class=PlainTextResponse)
def _lbcheck():
    return "OK"

@app.get("/health")
def health():
    sess = _load_onnx_session()
    return {
        "status": "ok",
        "onnxruntime": OS_HAS_ORT,
        "model_loaded": sess is not None,
        "model_path_exists": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH if os.path.exists(MODEL_PATH) else None,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "last_onnx_error": LAST_ONNX_ERROR,
    }

@app.get("/diag")
def diag():
    return {
        "SAVE_UPLOADS": SAVE_UPLOADS,
        "SAVE_HISTORY": SAVE_HISTORY,
        "uploads_dir": uploads_dir(),
        "history_len": len(load_history()),
        "onnxruntime": OS_HAS_ORT,
        "model_exists": os.path.exists(MODEL_PATH),
        "can_write_uploads": os.access(uploads_dir(), os.W_OK),
    }

# -----------------------------------------------------------------------------
# Analyze
# -----------------------------------------------------------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(...),
    project_id: Optional[str] = Form(default="PRJ-001"),
    column_id: Optional[str] = Form(default="K-01"),
    level: Optional[str] = Form(default=""),
    grid: Optional[str] = Form(default=""),
    location: Optional[str] = Form(default=""),
    meta: Optional[str] = Form(default=None),
):
    try:
        content = await image.read()
        if not content:
            raise HTTPException(status_code=400, detail="File kosong.")

        image_url = None
        if SAVE_UPLOADS:
            fn = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
            fullpath = os.path.join(uploads_dir(), fn)
            img = Image.open(io.BytesIO(content)).convert("RGB")
            img.save(fullpath, "JPEG", quality=88)
            image_url = f"/uploads/{fn}"

        meta_obj: Optional[ColumnMeta] = None
        if meta:
            try:
                meta_obj = ColumnMeta.model_validate_json(meta)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Invalid meta JSON: {e}")

        w, h, raw_dets, mask_ratios = run_inference(content)
        img_area = float(w * h)
        now = datetime.now().isoformat(timespec="seconds")

        items: List[DetectionItem] = []
        for idx, det in enumerate(raw_dets):
            x1, y1, x2, y2 = det["bbox"]
            if mask_ratios is not None and idx < len(mask_ratios):
                area_ratio = float(mask_ratios[idx])
            else:
                bbox_area = max(1, (x2 - x1)) * max(1, (y2 - y1))
                area_ratio = bbox_area / img_area

            raw_cls = det["cls_name"]
            dmg_type, sev_from_name = parse_type_and_severity(raw_cls)
            base_type = dmg_type if dmg_type in SUPPORTED_TYPES else "crack"
            sev_id = sev_from_name if sev_from_name in {"ringan","sedang","berat"} \
                     else infer_severity_by_ratio(base_type, area_ratio)
            severity_en = to_severity_en(sev_id)

            solution = solution_for(base_type, sev_id)
            keterangan = build_keterangan(base_type, sev_id)

            if raw_cls != base_type:
                raw_label_note = f"(model label: {raw_cls})"
                keterangan = (raw_label_note + "\n" + keterangan) if keterangan else raw_label_note

            pj = (project_id or (meta_obj.project_id if meta_obj else "") or "").strip()
            col = (column_id or (meta_obj.column_id if meta_obj else "") or "").strip()

            items.append(DetectionItem(
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                conf=float(det["conf"]),
                cls=base_type,
                severity=severity_en,
                solution=solution,
                keterangan=keterangan,
                project_id=pj,
                column_id=col,
                level=(level or (meta_obj.level if meta_obj else "") or ""),
                grid=(grid or (meta_obj.grid if meta_obj else "") or ""),
                location=(location or (meta_obj.location if meta_obj else "") or ""),
                timestamp=now,
                monitoring=Monitoring(status="pending", notes="", photos=[], updated_at=now),
            ))

        if SAVE_HISTORY:
            record = {
                "id": uuid.uuid4().hex,
                "project_id": project_id,
                "column_id": column_id,
                "timestamp": now,
                "image_url": image_url,
                "image_width": w,
                "image_height": h,
                "items": [i.model_dump(mode="json") for i in items],
            }
            if meta_obj:
                record["meta"] = meta_obj.model_dump(mode="json")
            append_history(record)

        return AnalyzeResponse(image_width=w, image_height=h, items=items)

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Analyze error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Export endpoints
# -----------------------------------------------------------------------------
@app.post("/export/csv")
async def export_csv(payload: ExportPayload):
    import csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["project_id","column_id","level","grid","location",
                     "damage_type","severity","confidence","bbox","timestamp","keterangan"])
    for it in payload.items:
        dmg_type, _ = parse_type_and_severity(it.cls.lower())
        writer.writerow([
            payload.project_id, it.column_id, it.level or "", it.grid or "", it.location or "",
            dmg_type, it.severity, f"{it.conf:.4f}", json.dumps(it.bbox, ensure_ascii=False),
            it.timestamp, (it.keterangan or "").replace("\n"," | ")
        ])
    mem = io.BytesIO(buf.getvalue().encode("utf-8")); mem.seek(0)
    return StreamingResponse(mem, media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="export_damage.csv"'})

@app.post("/export/json")
async def export_json(payload: ExportPayload):
    mem = io.BytesIO(json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=False).encode("utf-8")); mem.seek(0)
    return StreamingResponse(mem, media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="export_damage.json"'})

@app.post("/export/ifc-overlay")
async def export_ifc_overlay(payload: ExportPayload):
    overlay = {
        "project_id": payload.project_id,
        "pset_name": "Pset_Damage",
        "columns": [
            {
                "column_id": it.column_id,
                "props": {
                    "DamageType": parse_type_and_severity(it.cls)[0],
                    "DamageSeverity": it.severity,
                    "Confidence": round(it.conf, 4),
                    "Level": it.level or "",
                    "Grid": it.grid or "",
                    "Location": it.location or "",
                    "Timestamp": it.timestamp,
                    "Notes": it.keterangan,
                },
            } for it in payload.items
        ],
    }
    mem = io.BytesIO(json.dumps(overlay, indent=2, ensure_ascii=False).encode("utf-8")); mem.seek(0)
    return StreamingResponse(mem, media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="damage_ifc_overlay.json"'})

@app.post("/export/pdf")
async def export_pdf(payload: ExportPayload):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
    except Exception:
        raise HTTPException(status_code=501, detail="ReportLab tidak terpasang di server.")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 20*mm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20*mm, y, f"Damage Report — Project {payload.project_id}")
    y -= 8*mm
    c.setFont("Helvetica", 10)
    for it in payload.items:
        lines = [
            f"[{it.timestamp}] {it.column_id} — {it.cls} — {it.severity} — conf {it.conf:.2f}",
            f"Level: {it.level or '-'} | Grid: {it.grid or '-'} | Loc: {it.location or '-'}",
            f"bbox: {it.bbox}",
            f"Solusi: {it.solution}",
        ]
        for ln in lines:
            c.drawString(20*mm, y, ln); y -= 6*mm
            if y < 25*mm: c.showPage(); y = h - 20*mm; c.setFont("Helvetica", 10)
        for ln in (it.keterangan or "").split("\n"):
            c.drawString(20*mm, y, ln); y -= 5*mm
            if y < 25*mm: c.showPage(); y = h - 20*mm; c.setFont("Helvetica", 10)
        y -= 4*mm
        if y < 25*mm: c.showPage(); y = h - 20*mm; c.setFont("Helvetica", 10)
    c.save()
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="damage_report.pdf"'})

# -----------------------------------------------------------------------------
# History listing + Monitoring update + Delete
# -----------------------------------------------------------------------------
@app.get("/history")
def list_history(project_id: Optional[str] = Query(default=None),
                 column_id: Optional[str] = Query(default=None),
                 limit: int = Query(default=100, ge=1, le=1000)):
    data = load_history()
    out = []
    for r in data:
        if project_id and r.get("project_id") != project_id: continue
        if column_id and r.get("column_id") != column_id: continue
        out.append(r)
        if len(out) >= limit: break
    return out

@app.post("/history/monitor")
async def update_monitoring_endpoint(
    record_id: str = Form(...),
    item_index: int = Form(...),
    status: Optional[Literal["pending","in_progress","fixed"]] = Form(default=None),
    notes: Optional[str] = Form(default=None),
    photos: Optional[List[UploadFile]] = File(default=None),
):
    def _updater(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "items" not in rec or not isinstance(rec["items"], list):
            return rec
        if item_index < 0 or item_index >= len(rec["items"]):
            return rec
        item = rec["items"][item_index]
        mon = item.get("monitoring") or {}
        mon.setdefault("status", "pending")
        mon.setdefault("notes", "")
        mon.setdefault("photos", [])
        if status is not None:
            mon["status"] = status
        if notes is not None:
            mon["notes"] = notes
        if photos:
            for f in photos:
                try:
                    ext = os.path.splitext(f.filename or "")[1].lower() or ".jpg"
                    fn = f"fix_{record_id}_{item_index}_{uuid.uuid4().hex[:6]}{ext}"
                    path = os.path.join(uploads_dir(), fn)
                    content = f.file.read()
                    img = Image.open(io.BytesIO(content)).convert("RGB")
                    img.save(path, "JPEG", quality=88)
                    mon["photos"].append(f"/uploads/{fn}")
                except Exception:
                    continue
        mon["updated_at"] = datetime.now().isoformat(timespec="seconds")
        item["monitoring"] = mon
        rec["items"][item_index] = item
        return rec

    updated = update_history_record(record_id, _updater)
    if updated is None:
        raise HTTPException(status_code=404, detail="Record atau item tidak ditemukan.")
    return JSONResponse(updated)

# ---- DELETE per record
@app.delete("/history/{record_id}")
def delete_history_one(record_id: str):
    ok = delete_history_record(record_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Record tidak ditemukan.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# ---- DELETE bulk by filter (project_id/column_id). WAJIB salah satu diisi.
@app.delete("/history")
def delete_history_bulk(
    project_id: Optional[str] = Query(default=None),
    column_id: Optional[str] = Query(default=None),
):
    if not project_id and not column_id:
        raise HTTPException(status_code=400, detail="Harus menyertakan project_id atau column_id untuk penghapusan terfilter.")
    deleted = delete_history_filtered(project_id=project_id, column_id=column_id)
    return {"deleted": deleted}

# -----------------------------------------------------------------------------
# Serve React build (SPA)
# -----------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIST = os.path.join(ROOT_DIR, "frontend", "dist")
if os.path.isdir(FRONTEND_DIST):
    log.info("Serving React build from: %s", FRONTEND_DIST)
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/", response_class=PlainTextResponse)
    def index_text():
        return "Backend is up. FRONTEND_DIST not found."

if __name__ == "__main__":
    import uvicorn
    log.info("Starting server (onnx=%s, conf>=%.2f)", MODEL_PATH, CONF_THRESHOLD)
    uvicorn.run(app, host="0.0.0.0", port=8000)

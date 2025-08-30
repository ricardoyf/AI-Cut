# main.py - v4: 1s windows @ 3 fps + 720px + defectos de composición (texto, occlusión, clutter, tilt, saliencia)
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Forzar GPU global y optimizar backend de PyTorch ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from ultralytics import YOLO

# --- Logger Opcional ---
try:
    from ultralogger import Logger
    LOGGER = Logger()
except ImportError:
    class _L:
        def info(self, *a, **k): print("ℹ️", *a)
        def warning(self, *a, **k): print("⚠️", *a)
        def error(self, *a, **k): print("❌", *a)
        def success(self, *a, **k): print("✅", *a)
    LOGGER = _L()

# ==========================
# Configuración Global
# ==========================
TRAINING_DIR = Path(r"C:\ai-entrenamiento")
DATASET_DIR = TRAINING_DIR / "dataset"
LABELS_DIR = TRAINING_DIR / "labels"
MODELS_DIR = TRAINING_DIR / "models"
METRICS_PATH = TRAINING_DIR / "metrics.json"
SUPPORTED_VIDEO_EXTS = ['.mp4', '.mov', '.mkv', '.avi', '.webm']

# --- Segmentación uniforme ---
USE_UNIFORM_SEGMENTS = True    # activa la segmentación fija
UNIFORM_SEGMENT_SEC  = 0.5     # tamaño de segmento

# --- Tiempo / muestreo ---
ANALYSIS_SIDE = 720                 # ↑ detalle para defectos sutiles
MIN_SCENE_LEN_SEC = 1.0
WINDOW_DURATION_SEC = 1.0           # 1 ventana por segundo
WINDOW_OVERLAP_SEC = 0.0            # sin solape extra (ya vamos a 1Hz)
FRAMES_PER_WINDOW = 3               # t, t+0.33, t+0.66
FRAME_OFFSETS = [0.0, 0.33, 0.66]

# --- Detección / filtros básicos ---
YOLO_WEIGHTS = "yolov8n.pt"
DARK_LUMA_THR = 0.10

# --- Postproceso / reglas LC ---
MIN_DUR_FINAL = 2.0
FUSION_GAP = 0.30
PADDING_SEC = 0.10

# --- Selección por score ---
SCORE_MIN = 0.55
SCORE_Q = 0.70   # percentil dinámico (top 30%)

# --- Matching robusto (ingesta) ---
SCENE_MATCH_PAD_SEC = 0.15
SCENE_PAD_FOR_MATCH = 0.10

@dataclass
class Scene:
    start: float
    end: float
    idx: int

def _video_duration_cv2(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n   = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0.0 or n <= 0.0:
        return 0.0
    return float(n / fps)

# ==========================
# Utilidades
# ==========================
def _now_stamp() -> str: return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def _cache_paths(video_path: Path) -> dict:
    base = video_path.stem
    cache_dir = video_path.parent / ".aicache"
    cache_dir.mkdir(exist_ok=True)
    return {
        "samples": cache_dir / f"{base}.samples.parquet",
        "scenes": cache_dir / f"{base}.scenes.parquet",
        "meta": cache_dir / f"{base}.meta.json"
    }

def _elegidos_path(video_path: Path) -> Path: return video_path.with_name(f"{video_path.name}.csv")
def _propuestos_path(video_path: Path) -> Path: return video_path.with_name(f"{video_path.stem}.propuestos.csv")

def _write_propuestos_csv(path: Path, segments: list[tuple[float, float]]):
    path.write_text("".join([f"{s:.6f},{e:.6f},\n" for s, e in segments]), encoding="utf-8")

def _get_active_model_path() -> Path | None:
    MODELS_DIR.mkdir(exist_ok=True)
    models = list(MODELS_DIR.glob("modelo_xgb_v*.json"))
    if not models: return None
    models.sort(key=lambda p: int(p.stem.split('_v')[-1]))
    return models[-1]

def _resize_to_side(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) <= side: return img
    if h < w:
        new_h, new_w = side, int(w * side / h)
    else:
        new_w, new_h = side, int(h * side / w)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def _var_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _exposure_metrics(bgr_small: np.ndarray) -> Dict[str, float]:
    rgb = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    y = 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]
    return {
        "mean_luma": float(np.mean(y)),
        "p01": float(np.quantile(y, 0.01)),
        "p99": float(np.quantile(y, 0.99)),
        "clip_black": float(np.mean(y <= 0.01)),
        "clip_white": float(np.mean(y >= 0.99)),
    }

def _seek_ms(cap: cv2.VideoCapture, ms: float) -> bool:
    return cap.set(cv2.CAP_PROP_POS_MSEC, max(ms, 0.0))

def _read_gray_small(cap: cv2.VideoCapture) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    ok, frame = cap.read()
    if not ok or frame is None: return None, None
    small = _resize_to_side(frame, ANALYSIS_SIDE)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small, gray

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = max(1e-6, area_a + area_b - inter)
    return float(inter / union)

# ==========================
# Detectores / Extractores
# ==========================
class PersonDetector:
    def __init__(self, weights: str):
        if not torch.cuda.is_available():
            raise RuntimeError("🚫 No hay GPU de NVIDIA disponible.")
        self.model = YOLO(weights).to("cuda")
        self.half = True
        LOGGER.info("✅ YOLO cargado en GPU (FP16).")

    @torch.no_grad()
    def detect(self, img_bgr_small: np.ndarray):
        rgb = cv2.cvtColor(img_bgr_small, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        imgsz = min(1280, ((max(h, w) + 31)//32)*32)  # un poco más alto para personas pequeñas
        res = self.model.predict(rgb, verbose=False, imgsz=imgsz, device=0, half=self.half)
        boxes = []
        if res and res[0].boxes is not None:
            b = res[0].boxes.xyxy.detach().cpu().numpy()
            c = res[0].boxes.cls.detach().cpu().numpy().astype(int)
            for (x1, y1, x2, y2), ci in zip(b, c):
                boxes.append((float(x1), float(y1), float(x2), float(y2), int(ci)))
        persons = sum(1 for _,_,_,_,ci in boxes if ci == 0)
        return persons, boxes

class DefectFeatureExtractor:
    """Features de composición que penalizan escenas 'estropeadas'."""
    # Mapeo COCO (subset útil)
    COCO = [ 'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
             'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
             'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
             'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
             'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
             'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
             'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
             'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
             'remote','keyboard','cell phone','microwave','oven','toaster','sink',
             'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush' ]

    def __init__(self):
        # MSER para texto/overlay
        try:
            self.mser = cv2.MSER_create(_delta=5, _min_area=30, _max_area=5000)
        except Exception:
            self.mser = None
        # Saliency (degradación si no existe)
        try:
            self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        except Exception:
            self.saliency = None

    # --- Texto / overlay (MSER) ---
    def _text_coverage(self, bgr: np.ndarray, roi: Optional[Tuple[int,int,int,int]]=None) -> float:
        if self.mser is None:
            return 0.0
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        x1, y1, x2, y2 = (0, int(h*0.70), w, h) if roi is None else roi
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(w,x2), min(h,y2)
        if x2 <= x1 or y2 <= y1: return 0.0
        crop = gray[y1:y2, x1:x2]
        try:
            regions, _ = self.mser.detectRegions(crop)
        except Exception:
            return 0.0
        area = (y2-y1)*(x2-x1)
        covered = 0
        for cnt in regions:
            x,y,wc,hc = cv2.boundingRect(cnt)
            if wc*hc < 25: continue
            aspect = wc / max(1.0, hc)
            # cajas con aspecto de texto
            if 1.5 <= aspect <= 12.0:
                covered += wc*hc
        return float(min(1.0, covered / max(1.0, area)))

    # --- Clutter / ruido fuera del sujeto ---
    def _clutter_score(self, gray: np.ndarray, subject_boxes: List[Tuple[float,float,float,float]]) -> float:
        edges = cv2.Canny(gray, 80, 160)
        mask = np.zeros_like(edges, np.uint8)
        for (x1,y1,x2,y2) in subject_boxes:
            cv2.rectangle(mask, (int(x1),int(y1)), (int(x2),int(y2)), 255, -1)
        bg_edges = np.where(mask==255, 0, edges)
        # densidad de bordes + entropía aproximada
        density = float(np.mean(bg_edges>0))
        hist = cv2.calcHist([bg_edges],[0],None,[8],[0,256]).ravel()
        p = hist / max(1e-6, np.sum(hist))
        ent = -np.sum(p*np.log(p+1e-12)) / np.log(len(p))
        return float(0.7*density + 0.3*ent)

    # --- Horizonte inclinado (líneas) ---
    def _tilt_degrees(self, gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 80, 160)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
        if lines is None or len(lines)==0:
            # fallback: gradiente dominante
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            ang = (np.arctan2(gy, gx) * 180.0 / np.pi).ravel()
            if ang.size == 0: return 0.0
            mean = float(np.mean(np.abs(((ang+90)%180)-90)))  # cercanía a 0/90
            return mean
        # ángulo medio en grados mod 90
        angles = []
        for rho, theta in lines[:,0]:
            deg = theta * 180.0 / np.pi
            d = abs(((deg+45)%90)-45)  # distancia a {0,90}
            angles.append(d)
        return float(np.mean(angles)) if angles else 0.0

    # --- Sujeto cortado por borde ---
    def _subject_cut(self, person_box: Optional[Tuple[float,float,float,float]], hw: Tuple[int,int]) -> float:
        if person_box is None: return 0.0
        x1,y1,x2,y2 = person_box
        h,w = hw
        m = 4  # margen en px (imagen ya reducida)
        cut = int(x1<=m) + int(y1<=m) + int(x2>=w-m) + int(y2>=h-m)
        return float(min(1, cut))

    # --- Saliencia fuera de puntos de tercios ---
    def _saliency_off_thirds(self, bgr: np.ndarray, boxes: List[Tuple[float,float,float,float,int]]) -> float:
        h, w = bgr.shape[:2]
        if self.saliency is None:
            # fallback: centroide de intensidad como "saliencia"
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            ys, xs = np.mgrid[0:h, 0:w]
            total = np.sum(g)+1e-6
            cx = float(np.sum(xs*g)/total); cy = float(np.sum(ys*g)/total)
        else:
            ok, sal = self.saliency.computeSaliency(bgr)
            if not ok: return 0.0
            sal = (sal * 255).astype(np.uint8)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(sal)
            cx, cy = float(maxLoc[0]), float(maxLoc[1])

        ppts = [(w/3,h/3),(2*w/3,h/3),(w/3,2*h/3),(2*w/3,2*h/3)]
        dmin = min(np.hypot(cx-px, cy-py) for (px,py) in ppts)
        # normalizar por la diagonal
        norm = np.hypot(w, h)
        return float(min(1.0, dmin / (0.35*norm)))  # >1 ⇒ lejano; cap en 1

    # --- Occlusión cara/cuerpo con objetos ---
    def _occlusion_flag(self, person_box: Optional[Tuple[float,float,float,float]], other_boxes: List[Tuple[float,float,float,float]], thr=0.15) -> float:
        if person_box is None or not other_boxes: return 0.0
        px1,py1,px2,py2 = person_box
        p_area = max(1e-6, (px2-px1)*(py2-py1))
        occ_area = 0.0
        for bx1,by1,bx2,by2 in other_boxes:
            # área de intersección con persona
            inter = _iou((px1,py1,px2,py2),(bx1,by1,bx2,by2)) * p_area
            occ_area += inter
        ratio = occ_area / p_area
        return float(ratio >= thr)

    def extract(self, bgr_small: np.ndarray, gray: np.ndarray, boxes: List[Tuple[float,float,float,float,int]]) -> dict:
        h, w = gray.shape[:2]
        person_boxes = [(x1,y1,x2,y2) for (x1,y1,x2,y2,ci) in boxes if self.COCO[ci]=='person']
        main_person = None
        if person_boxes:
            main_person = max(person_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        other_boxes = [(x1,y1,x2,y2) for (x1,y1,x2,y2,ci) in boxes if self.COCO[ci] != 'person']

        # texto: bottom band + sobre rostro (si existe)
        text_cov_bottom = self._text_coverage(bgr_small, None)
        text_cov_face = 0.0
        if main_person is not None:
            x1,y1,x2,y2 = map(int, main_person)
            x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(w,x2), min(h,y2)
            text_cov_face = self._text_coverage(bgr_small, (x1,y1,x2,y2))

        # clutter fuera del sujeto
        clutter = self._clutter_score(gray, person_boxes)

        # tilt (horizonte inclinado)
        tilt = self._tilt_degrees(gray)

        # sujeto cortado
        subj_cut = self._subject_cut(main_person, (h,w))

        # saliencia fuera de tercios
        sal_off = self._saliency_off_thirds(bgr_small, boxes)

        # occlusión
        occ = self._occlusion_flag(main_person, other_boxes)

        return dict(
            text_cover_bottom=text_cov_bottom,
            text_cover_face=text_cov_face,
            clutter_score=clutter,
            tilt_deg=tilt,
            subject_cut_flag=subj_cut,
            saliency_off_thirds=sal_off,
            occlusion_flag=occ,
        )

# ==========================
# Detección de escenas y sampling
# ==========================
def detect_scenes(video_path: str) -> List[Scene]:
    """
    Si USE_UNIFORM_SEGMENTS=True -> divide el vídeo en segmentos uniformes de UNIFORM_SEGMENT_SEC.
    Si False -> fallback a PySceneDetect (tu lógica original).
    """
    if USE_UNIFORM_SEGMENTS:
        try:
            dur = _video_duration_cv2(video_path)
            if dur <= 0.0:
                LOGGER.error(f"No se pudo obtener la duración de {Path(video_path).name}")
                return []
            scenes: List[Scene] = []
            idx = 0
            t = 0.0
            step = float(UNIFORM_SEGMENT_SEC)
            eps = 1e-9
            while t + eps < dur:
                end = min(dur, t + step)
                # Importante: NO filtramos por longitud aquí (queremos 0.5 s)
                scenes.append(Scene(start=float(t), end=float(end), idx=idx))
                idx += 1
                t += step
            if not scenes:
                scenes = [Scene(start=0.0, end=float(dur), idx=0)]
            return scenes
        except Exception as e:
            LOGGER.error(f"Error segmentando uniformemente {video_path}: {e}")
            return []

    # --- Fallback: PySceneDetect (tu código original) ---
    try:
        video = open_video(video_path)
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=22))
        sm.detect_scenes(video, show_progress=False)
        scene_list = sm.get_scene_list()
        scenes: List[Scene] = []
        for i, (s, e) in enumerate(scene_list):
            start, end = s.get_seconds(), e.get_seconds()
            if end - start >= MIN_SCENE_LEN_SEC:
                scenes.append(Scene(start=float(start), end=float(end), idx=i))
        if not scenes and video.duration > 0:
            scenes.append(Scene(start=0.0, end=float(video.duration), idx=0))
        return scenes
    except Exception as e:
        LOGGER.error(f"Error detectando escenas en {video_path}: {e}")
        return []

def sample_times_within_scene(start: float, end: float) -> List[float]:
    if end - start < WINDOW_DURATION_SEC: return [start]
    times = []
    t = start
    step = WINDOW_DURATION_SEC - WINDOW_OVERLAP_SEC
    while t + WINDOW_DURATION_SEC <= end + 1e-6:
        times.append(round(t, 3))
        t += step
    if not times: times = [start]
    return times

# ==========================
# Análisis principal
# ==========================
def analyze_hybrid(video_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")

    pdtr = PersonDetector(YOLO_WEIGHTS)
    defect = DefectFeatureExtractor()
    scenes = detect_scenes(video_path)

    records = []
    pbar = tqdm(scenes, desc="Analizando segmentos 0.5s" if USE_UNIFORM_SEGMENTS else "Analizando escenas")
    for sc in pbar:
        times = sample_times_within_scene(sc.start, sc.end)
        for t_start in times:
            # leer 3 frames dentro de la ventana de 1s
            per_window_feats = []
            persons_list, boxes_list, sharps, expos = [], [], [], []

            for off in FRAME_OFFSETS:
                tt = min(sc.end - 1e-3, t_start + off)
                _seek_ms(cap, tt * 1000.0)
                small, gray = _read_gray_small(cap)
                if small is None: continue
                persons, boxes = pdtr.detect(small)
                expo = _exposure_metrics(small)
                if expo["mean_luma"] < DARK_LUMA_THR:
                    continue  # descarta frames demasiado oscuros

                feat = defect.extract(small, gray, boxes)
                per_window_feats.append(feat)
                persons_list.append(persons)
                boxes_list.append(boxes)
                sharps.append(_var_laplacian(gray))
                expos.append(expo)

            if not per_window_feats:
                continue

            # agregación por ventana (medias para continuas, max en flags/coberturas)
            def mean_key(k, default=0.0):
                vals = [f[k] for f in per_window_feats if k in f]
                return float(np.mean(vals)) if vals else default
            def max_key(k, default=0.0):
                vals = [f[k] for f in per_window_feats if k in f]
                return float(np.max(vals)) if vals else default

            rec = dict(
                video=Path(video_path).name,
                scene_idx=sc.idx,
                scene_start=sc.start,
                scene_end=sc.end,
                t=t_start,
                persons=max(persons_list) if persons_list else 0,
                sharpness=float(np.mean(sharps)) if sharps else 0.0,
                mean_luma=float(np.mean([e["mean_luma"] for e in expos])) if expos else 0.0,
                p01=float(np.mean([e["p01"] for e in expos])) if expos else 0.0,
                p99=float(np.mean([e["p99"] for e in expos])) if expos else 0.0,
                clip_black=float(np.mean([e["clip_black"] for e in expos])) if expos else 0.0,
                clip_white=float(np.mean([e["clip_white"] for e in expos])) if expos else 0.0,
                # defectos agregados
                text_cover_bottom_mean=mean_key("text_cover_bottom"),
                text_cover_face_max=max_key("text_cover_face"),
                clutter_score_mean=mean_key("clutter_score"),
                tilt_deg_mean=mean_key("tilt_deg"),
                subject_cut_flag_max=max_key("subject_cut_flag"),
                saliency_off_thirds_mean=mean_key("saliency_off_thirds"),
                occlusion_flag_max=max_key("occlusion_flag"),
            )

            records.append(rec)

    cap.release()
    if not records:
        return (pd.DataFrame(), pd.DataFrame())

    df_samples = pd.DataFrame.from_records(records)

    # --- Agregación por escena ---
    agg_spec = {
        # base
        "persons": ["max", "mean"],
        "sharpness": ["mean", "max"],
        "mean_luma": ["mean"],
        "p01": ["mean"],
        "p99": ["mean"],
        "clip_black": ["mean"],
        "clip_white": ["mean"],
        # defectos
        "text_cover_bottom_mean": ["mean", "max"],
        "text_cover_face_max": ["max"],
        "clutter_score_mean": ["mean", "max"],
        "tilt_deg_mean": ["mean", "max"],
        "subject_cut_flag_max": ["max"],
        "saliency_off_thirds_mean": ["mean", "max"],
        "occlusion_flag_max": ["max"],
    }

    df_scenes = df_samples.groupby(
        ["video", "scene_idx", "scene_start", "scene_end"], as_index=False
    ).agg(agg_spec)

    # aplanar nombres
    df_scenes.columns = ["_".join(col).strip() if isinstance(col, tuple) and col[1] else col[0] for col in df_scenes.columns]
    for col in ("scene_start", "scene_end"):
        if col in df_scenes.columns:
            df_scenes[col] = df_scenes[col].astype(float)

    return df_samples, df_scenes

# ==========================
# Modelo y Puntuación
# ==========================
MODEL_FEATURES = [
    # Base
    'persons_max','persons_mean','sharpness_mean','sharpness_max','mean_luma_mean','p01_mean','p99_mean','clip_black_mean','clip_white_mean',
    # Defectos / composición
    'text_cover_bottom_mean_mean','text_cover_bottom_mean_max','text_cover_face_max_max',
    'clutter_score_mean_mean','clutter_score_mean_max',
    'tilt_deg_mean_mean','tilt_deg_mean_max',
    'subject_cut_flag_max_max',
    'saliency_off_thirds_mean_mean','saliency_off_thirds_mean_max',
    'occlusion_flag_max_max',
]

def _score_with_model(df_scenes: pd.DataFrame, model_path: Path) -> list[float]:
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    try:
        model.set_params(tree_method="hist", device="cuda")
    except Exception:
        try:
            model.set_params(tree_method="gpu_hist")
        except Exception:
            pass

    # Asegurar columnas
    X_df = df_scenes.copy()
    for col in MODEL_FEATURES:
        if col not in X_df.columns:
            X_df[col] = 0.0

    X = X_df[MODEL_FEATURES].astype(float).fillna(0.0)
    return model.predict_proba(X)[:, 1].tolist()

def _score_stub(row: dict) -> float:
    # Fallback simple: personas/nitidez favorecen, defectos penalizan
    score = 0.0
    score += 0.8 * row.get("persons_max", 0.0)
    score += 0.001 * row.get("sharpness_mean", 0.0)
    penalty = (
        1.2*row.get("text_cover_face_max_max", 0.0) +
        0.8*row.get("text_cover_bottom_mean_mean", 0.0) +
        0.8*row.get("clutter_score_mean_mean", 0.0) +
        0.6*row.get("saliency_off_thirds_mean_mean", 0.0) +
        0.6*row.get("occlusion_flag_max_max", 0.0) +
        0.5*(row.get("tilt_deg_mean_mean", 0.0)/15.0) +
        0.8*row.get("subject_cut_flag_max_max", 0.0)
    )
    return float(score - penalty)

def _rank_scenes(df: pd.DataFrame, scores: list[float] | None = None) -> list[tuple[float, float, float]]:
    if scores is None:
        scores = [_score_stub(r) for r in df.to_dict("records")]
    df['score'] = scores
    return [(float(r["scene_start"]), float(r["scene_end"]), float(r["score"])) for r in df.to_dict("records")]

def _postprocess(segments: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    segs = [(s, e, sc) for s, e, sc in segments if e - s >= MIN_DUR_FINAL]
    if not segs: return []
    scores = [sc for _, _, sc in segs]
    thr_q = float(np.quantile(scores, SCORE_Q))
    thr = max(SCORE_MIN, thr_q)
    segs_filtered = [(s, e, sc) for s, e, sc in segs if sc >= thr]
    if not segs_filtered:
        s, e, sc = max(segments, key=lambda x: x[2])
        segs_filtered = [(s, e, sc)]
    segs_filtered.sort(key=lambda x: (x[0], -x[2]))
    merged = []
    for s, e, sc in segs_filtered:
        if not merged or s - merged[-1][1] >= FUSION_GAP:
            merged.append((s, e, sc))
        else:
            ps, pe, psc = merged[-1]
            merged[-1] = (ps, max(pe, e), max(psc, sc))
    return [(max(0.0, s - PADDING_SEC), e + PADDING_SEC) for s, e, _ in sorted(merged, key=lambda x: x[2], reverse=True)]

def _read_elegidos_csv(csv_path: Path) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for line in csv_path.read_text(encoding="utf-8").splitlines()
            if (p := line.strip().rstrip(",").split(",")) and len(p) >= 2]

def _overlap_len(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

def _label_scenes_with_weights(scenes_df: pd.DataFrame, positivos: list[tuple[float, float]]):
    labels, weights = [], []
    pos_exp = [(max(0.0, s - SCENE_MATCH_PAD_SEC), e + SCENE_MATCH_PAD_SEC) for s, e in positivos]
    for _, row in scenes_df.iterrows():
        s, e = float(row['scene_start']), float(row['scene_end'])
        s_m, e_m = max(0.0, s - SCENE_PAD_FOR_MATCH), e + SCENE_PAD_FOR_MATCH
        dur_m = max(0.0, e_m - s_m)
        lab, w = 0, 0.2
        if dur_m > 0:
            best_metric = 0.0
            for gs, ge in pos_exp:
                overlap = _overlap_len((s_m, e_m), (gs, ge))
                elegido_dur = max(0.0, ge - gs)
                union = max(e_m, ge) - min(s_m, gs)
                cov_min = overlap / max(1e-9, min(dur_m, elegido_dur))
                iou = overlap / max(1e-9, union)
                best_metric = max(best_metric, max(cov_min, iou))
            if best_metric >= 0.5: lab = 1
            w = max(0.2, min(1.0, best_metric))
        labels.append(lab); weights.append(w)
    return labels, weights

def _make_xgb():
    params = dict(objective='binary:logistic', eval_metric='logloss', n_estimators=400,
                  learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9)
    try:
        return xgb.XGBClassifier(**params, tree_method="hist", device="cuda")
    except Exception:
        try:
            return xgb.XGBClassifier(**params, tree_method="gpu_hist")
        except Exception:
            return xgb.XGBClassifier(**params)

def _fit_xgb_compat(model, X_train, y_train, X_val, y_val, sample_weight=None):
    try:
        from xgboost.callback import EarlyStopping
        return model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)],
                         verbose=False, callbacks=[EarlyStopping(rounds=30, save_best=True)])
    except ImportError:
        try:
            return model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)],
                             eval_metric="logloss", early_stopping_rounds=30, verbose=False)
        except TypeError:
            return model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)

# ==========================
# Handlers CLI
# ==========================
def handle_analyze(args):
    video_path = Path(args.video_path)
    force = getattr(args, "force", False)
    if not video_path.exists():
        LOGGER.error(f"Vídeo no existe: {video_path.name}")
        return False
    cache = _cache_paths(video_path)
    if cache["meta"].exists() and not force:
        LOGGER.info("Análisis cacheado existe. Usa --force para re-analizar.")
        return True
    LOGGER.info(f"Iniciando análisis para: {video_path.name}")
    try:
        if force:
            [p.unlink(missing_ok=True) for p in cache.values()]
        df_samples, df_scenes = analyze_hybrid(str(video_path))
        if not df_scenes.empty:
            df_scenes.to_parquet(cache["scenes"])
            df_samples.to_parquet(cache["samples"])
            meta = {"video": video_path.name, "ts": _now_stamp(), "scene_count": len(df_scenes)}
            cache["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
            LOGGER.success(f"Análisis completo para {video_path.name}. ({len(df_scenes)} escenas)")
        else:
            LOGGER.warning(f"No se generaron escenas para {video_path.name}.")
        return True
    except Exception as e:
        LOGGER.error(f"analizando {video_path.name}: {e}")
        return False

def handle_propose(args):
    video_path = Path(args.video_path)
    force = getattr(args, "force", False)
    if not video_path.exists():
        LOGGER.error(f"Vídeo no existe: {video_path.name}")
        return False
    scenes_path = _cache_paths(video_path)["scenes"]
    if not scenes_path.exists() or force:
        LOGGER.info(f"No hay análisis previo. Ejecutando 'analyze' para {video_path.name}...")
        if not handle_analyze(argparse.Namespace(video_path=str(video_path), force=True)):
            return False
    LOGGER.info(f"Generando propuestas para: {video_path.name}")
    try:
        df = pd.read_parquet(scenes_path) if scenes_path.exists() else pd.DataFrame()
        if df.empty:
            LOGGER.warning("No hay escenas para puntuar.")
            final_segments = []
        else:
            model_path = _get_active_model_path()
            scores = _score_with_model(df, model_path) if model_path else None
            if model_path:
                LOGGER.info(f"Usando modelo: {model_path.name}")
            else:
                LOGGER.warning("No hay modelo entrenado. Usando reglas básicas.")
            ranked = _rank_scenes(df, scores)
            final_segments = _postprocess(ranked)
        out_csv = _propuestos_path(video_path)
        _write_propuestos_csv(out_csv, final_segments)
        LOGGER.success(f"Exportado: {out_csv.name} ({len(final_segments)} segmentos)")
        return True
    except Exception as e:
        LOGGER.error(f"proponiendo para {video_path.name}: {e}")
        return False

def handle_ingest(args):
    video_path = Path(args.video_path)
    csv_path = _elegidos_path(video_path)
    if not video_path.exists():
        LOGGER.error(f"Vídeo no existe: {video_path.name}")
        return False
    if not csv_path.exists():
        LOGGER.error(f"CSV de etiquetas no encontrado: {csv_path.name}")
        return False
    if not handle_analyze(argparse.Namespace(video_path=str(video_path), force=True)):
        LOGGER.error(f"Falló el análisis de {video_path.name} durante la ingesta.")
        return False
    try:
        scenes_path = _cache_paths(video_path)["scenes"]
        if not scenes_path.exists():
            LOGGER.warning("No hay features para etiquetar.")
            return True
        df_scenes = pd.read_parquet(scenes_path)
        positivos = _read_elegidos_csv(csv_path)
        labels, weights = _label_scenes_with_weights(df_scenes, positivos)
        if len(labels) != len(df_scenes):
            LOGGER.error("Discrepancia etiquetas/features")
            return False
        df_scenes["label"] = labels
        df_scenes["weight"] = weights
        df_scenes["video"] = video_path.name

        dataset_path = DATASET_DIR / "dataset.parquet"
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        if dataset_path.exists():
            df_old = pd.read_parquet(dataset_path)
            df_final = pd.concat([df_old[df_old["video"] != video_path.name], df_scenes], ignore_index=True)
        else:
            df_final = df_scenes
        pq.write_table(pa.Table.from_pandas(df_final), dataset_path)

        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        archived = LABELS_DIR / f"{video_path.stem}_{_now_stamp()}.csv"
        shutil.copy2(csv_path, archived)
        LOGGER.success(f"Ingesta completada para {video_path.name} con lógica robusta.")
        return True
    except Exception as e:
        LOGGER.error(f"durante la ingesta de {video_path.name}: {e}")
        return False

def handle_train(args):
    LOGGER.info("Ejecutando comando 'train'.")
    dataset_path = DATASET_DIR / "dataset.parquet"
    if not dataset_path.exists():
        LOGGER.error("dataset.parquet no encontrado.")
        return
    df = pd.read_parquet(dataset_path)
    if 'weight' not in df.columns:
        LOGGER.warning("Columna 'weight' no existe. Se usarán pesos uniformes.")
        df['weight'] = 1.0

    LOGGER.info(f"Dataset cargado con {len(df)} escenas de {df['video'].nunique()} vídeos.")
    X = df.copy()
    y = df["label"]
    groups = df["video"]

    # asegurar columnas del modelo
    for col in MODEL_FEATURES:
        if col not in X.columns:
            X[col] = 0.0

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))
    X_train = X.iloc[train_idx][MODEL_FEATURES].astype(float).fillna(0.0)
    X_val   = X.iloc[val_idx][MODEL_FEATURES].astype(float).fillna(0.0)
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    w_train = df.iloc[train_idx]["weight"].astype(float).fillna(1.0)

    LOGGER.info(f"Split: {len(X_train)} train, {len(X_val)} val.")
    model = _make_xgb()
    _fit_xgb_compat(model, X_train, y_train, X_val, y_val, sample_weight=w_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    score = average_precision_score(y_val, y_pred_proba)
    LOGGER.success(f"Entrenamiento completo. AUPRC: {score:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    last_v = int(p.stem.split('_v')[-1]) if (p := _get_active_model_path()) else 0
    new_model_path = MODELS_DIR / f"modelo_xgb_v{last_v + 1}.json"
    model.save_model(new_model_path)
    LOGGER.info(f"Modelo guardado en: {new_model_path}")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    metrics[f"v{last_v + 1}"] = {"auprc": score, "ts": _now_stamp(), "train_scenes": len(X_train)}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

def handle_ingest_batch(args):
    target_dir = Path(args.target_folder)
    LOGGER.info(f"Iniciando ingesta batch desde: {target_dir}")
    video_files = [p for p in target_dir.glob('*') if p.suffix.lower() in SUPPORTED_VIDEO_EXTS]
    pairs = [v for v in video_files if _elegidos_path(v).exists()]
    LOGGER.info(f"Encontrados {len(pairs)} pares vídeo/csv para procesar.")
    if not pairs: return
    success = sum(1 for v in tqdm(pairs, desc="Ingestando Batch")
                  if handle_ingest(argparse.Namespace(video_path=str(v))))
    LOGGER.success(f"Ingesta Batch Completa: {success}/{len(pairs)} exitosas")

def handle_propose_batch(args):
    target_dir = Path(args.target_folder)
    LOGGER.info(f"Iniciando propuestas batch desde: {target_dir}")
    video_files = [p for p in target_dir.glob('*') if p.suffix.lower() in SUPPORTED_VIDEO_EXTS]
    LOGGER.info(f"Encontrados {len(video_files)} vídeos para procesar.")
    if not video_files: return
    success = sum(1 for v in tqdm(video_files, desc="Proponiendo Batch")
                  if handle_propose(argparse.Namespace(video_path=str(v), force=False)))
    LOGGER.success(f"Propuestas Batch Completas: {success}/{len(video_files)} exitosas")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-CUT: Asistente de edición de vídeo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_an = subparsers.add_parser("analyze", help="Analiza un vídeo")
    p_an.add_argument("video_path", type=str)
    p_an.add_argument("--force", action="store_true", help="Fuerza el re-analisis")
    p_an.set_defaults(func=handle_analyze)

    p_pr = subparsers.add_parser("propose", help="Propone escenas para un vídeo")
    p_pr.add_argument("video_path", type=str)
    p_pr.add_argument("--force", action="store_true", help="Fuerza un nuevo análisis")
    p_pr.set_defaults(func=handle_propose)

    p_pr_b = subparsers.add_parser("propose-batch", help="Propone escenas para todos los vídeos de una carpeta")
    p_pr_b.add_argument("target_folder", type=str)
    p_pr_b.set_defaults(func=handle_propose_batch)

    p_in = subparsers.add_parser("ingest", help="Ingiere un par <video>/<video.csv>")
    p_in.add_argument("video_path", type=str)
    p_in.set_defaults(func=handle_ingest)

    p_in_b = subparsers.add_parser("ingest-batch", help="Ingiere todos los pares vídeo/csv de una carpeta")
    p_in_b.add_argument("target_folder", type=str, nargs='?', default=str(TRAINING_DIR),
                        help=f"Carpeta a procesar (defecto: {TRAINING_DIR})")
    p_in_b.set_defaults(func=handle_ingest_batch)

    p_tr = subparsers.add_parser("train", help="Re-entrena el modelo de XGBoost")
    p_tr.set_defaults(func=handle_train)

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
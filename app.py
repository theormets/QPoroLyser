# Full corrected app.py — PorosAnalyser (final)
# - Uses the exact description text you provided (properly quoted)
# - Orange header matched to Analyze button
# - Centered image/controls, automatic Single & Multi runs, summary cards above annotated images
# - Keeps detection and geometry logic unchanged
# - Updated with publication-worthy visualizations and renamed labels

import os
import cv2
import math
import tempfile
import warnings
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import torch
from ultralytics import YOLO

# ---------------------------
# Model path resolution (Option B)
# ---------------------------
_cands = {
    "qual_single": ["models/qual_single.pt", "qual_single.pt"],
    "qual_multi":  ["models/qual_multi.pt",  "qual_multi.pt"],
    "quant_single":["models/quant_single.pt","quant_single.pt"],
    "quant_multi": ["models/quant_multi.pt", "quant_multi.pt"],
}

def resolve_model_path(slot):
    for p in _cands.get(slot, []):
        if os.path.exists(p):
            return p
    root_key = slot.split("_")[0]
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith(".pt") and root_key in f.lower():
                return os.path.join(root, f)
    return None

QUAL_SINGLE_PT  = resolve_model_path("qual_single")
QUAL_MULTI_PT   = resolve_model_path("qual_multi")
QUANT_SINGLE_PT = resolve_model_path("quant_single")
QUANT_MULTI_PT  = resolve_model_path("quant_multi")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Colors and Label Mappings
# ---------------------------
COLOR_SINGLE = (0, 0, 255)       # bright red for single-class
COLOR_MULTI_MAP = {
    "IP": (0, 0, 255),
    "NP": (0, 255, 0),
    "RP": (255, 0, 0),
}
PALETTE = [
    (0, 128, 255),
    (0, 255, 255),
    (255, 128, 0),
    (200, 0, 200),
    (0, 200, 200),
    (200, 200, 0),
]

# Publication-quality colors and label mapping
PLOT_COLORS = {"IP":"#d62728","NP":"#2ca02c","RP":"#1f77b4"}  # red, green, blue
LABEL_MAP = {
    "IP": "Keyhole (Irregular)",
    "NP": "Lack-of-Fusion (Network)", 
    "RP": "Gas (Round)"
}

# ---------------------------
# Model loader (cached)
# ---------------------------
_model_cache = {}
def load_model(path):
    if path in _model_cache:
        return _model_cache[path]
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = YOLO(path)
    try:
        model.to(DEVICE)
    except Exception:
        pass
    _model_cache[path] = model
    return model

# ---------------------------
# Helpers (I/O, geometry, plotting — unchanged)
# ---------------------------
def save_image_rgb(img_rgb):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t:
        cv2.imwrite(t.name, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return t.name

def dataframe_to_excel_bytes(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        df.to_excel(tmp.name, index=False)
        return tmp.name

def seg_to_pts(seg):
    try:
        arr = np.array(seg, dtype=float)
        if arr.ndim == 1:
            if len(arr) % 2 != 0:
                return None
            return arr.reshape(-1, 2)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]
    except Exception:
        return None
    return None

def short_label(long_name: str) -> str:
    if not long_name:
        return ""
    s = str(long_name).strip(); up = s.upper(); low = s.lower()
    if up in ("IP","NP","RP"):
        return up
    if "network" in low or "net" in low or "web" in low:
        return "NP"
    if "round" in low or "circular" in low or low.startswith("r"):
        return "RP"
    if "irregular" in low or "irreg" in low or low.startswith("i"):
        return "IP"
    fallback = "".join([c for c in up if c.isalnum()])[:2]
    return fallback if fallback else up[:2]

def polygon_area_shoelace(pts):
    if pts is None or len(pts) < 3:
        return 0.0
    x = pts[:,0]; y = pts[:,1]
    s1 = np.dot(x, np.roll(y, -1))
    s2 = np.dot(y, np.roll(x, -1))
    area = 0.5 * abs(s1 - s2)
    return float(area)

def polygon_perimeter(pts):
    if pts is None or len(pts) < 2:
        return 0.0
    diffs = np.diff(np.vstack([pts, pts[0]]), axis=0)
    seg_lengths = np.hypot(diffs[:,0], diffs[:,1])
    return float(seg_lengths.sum())

def polygon_centroid(pts):
    if pts is None or len(pts) == 0:
        return (None, None)
    A = polygon_area_shoelace(pts)
    if A == 0:
        return (float(pts[:,0].mean()), float(pts[:,1].mean()))
    x = pts[:,0]; y = pts[:,1]
    xi = x; yi = y; xi1 = np.roll(x, -1); yi1 = np.roll(y, -1)
    cross = xi * yi1 - xi1 * yi
    Cx = (1.0/(6.0*A)) * np.sum((xi + xi1) * cross)
    Cy = (1.0/(6.0*A)) * np.sum((yi + yi1) * cross)
    return (float(Cx), float(Cy))

def polygon_bbox(pts):
    if pts is None or len(pts) == 0:
        return (None, None, None, None)
    minx = float(np.min(pts[:,0])); miny = float(np.min(pts[:,1]))
    maxx = float(np.max(pts[:,0])); maxy = float(np.max(pts[:,1]))
    width = maxx - minx; height = maxy - miny
    return (minx, miny, width, height)

def convex_hull_area(pts):
    if pts is None or len(pts) < 3:
        return 0.0
    pts_int = pts.reshape(-1,1,2).astype(np.int32)
    hull = cv2.convexHull(pts_int)
    return float(cv2.contourArea(hull))

def bar_chart_area(area_dict):
    """Publication-quality bar chart with renamed labels"""
    classes = ["RP","NP","IP"]
    display_labels = [LABEL_MAP.get(c, c) for c in classes]
    values = [sum(area_dict.get(c,[])) for c in classes]
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = [PLOT_COLORS.get(c,"gray") for c in classes]
    
    bars = ax.bar(display_labels, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax.set_xlabel("Defect Type", fontsize=14, fontweight='bold')
    ax.set_ylabel("Total Area (px²)", fontsize=14, fontweight='bold')
    ax.set_title("Total Area by Defect Type", fontsize=16, fontweight='bold', pad=20)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t:
        fig.savefig(t.name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return t.name

def hist_single(areas):
    """Publication-quality histogram for single-class"""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    n, bins, patches = ax.hist(areas, bins=20, color="#d62728", edgecolor="black", 
                                linewidth=1.2, alpha=0.85)
    
    ax.set_xlabel("Area (px²)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=14, fontweight='bold')
    ax.set_title("Pore Area Distribution", fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add statistics text box
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    textstr = f'Mean: {mean_area:.1f} px²\nStd: {std_area:.1f} px²\nN: {len(areas)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t:
        fig.savefig(t.name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return t.name

def create_count_visualization(mapped_counts):
    """Create a publication-quality bar chart (histogram) for multi-class counts - the 'Fingerprint'"""
    classes = ["RP", "NP", "IP"]
    display_labels = [LABEL_MAP.get(c, c) for c in classes]
    values = [mapped_counts.get(c, 0) for c in classes]
    colors = [PLOT_COLORS.get(c, "gray") for c in classes]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Create bar chart
    bars = ax.bar(display_labels, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax.set_xlabel("Defect Type", fontsize=14, fontweight='bold')
    ax.set_ylabel("Count", fontsize=14, fontweight='bold')
    ax.set_title("Defect Fingerprint: Distribution by Count", fontsize=16, fontweight='bold', pad=20)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=11)
    plt.xticks(rotation=15, ha='right')
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t:
        fig.savefig(t.name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return t.name

def _get_conf_for_index(r, i):
    try:
        if hasattr(r, "boxes") and hasattr(r.boxes, "conf"):
            confs = r.boxes.conf.cpu().numpy()
            if i < len(confs):
                return float(confs[i])
    except Exception:
        pass
    try:
        masks = getattr(r, "masks", None)
        if masks is not None:
            for attr in ("conf", "probs"):
                if hasattr(masks, attr):
                    arr = getattr(masks, attr)
                    try:
                        arr_np = np.array(arr)
                        if i < len(arr_np):
                            return float(arr_np[i])
                    except Exception:
                        pass
    except Exception:
        pass
    return 0.0

def draw_label_once(img_bgr, text, pos, color_bgr, drawn_positions, radius=6, font_scale=0.45, thickness=1, outline=True):
    if pos is None:
        return
    try:
        x0, y0 = int(pos[0]), int(pos[1])
    except Exception:
        return

    h, w = img_bgr.shape[:2]
    offsets = [(0, 0), (8, 0), (-8, 0), (0, 8), (0, -8), (8, 8), (-8, -8), (12, 0), (0, 12)]
    chosen = None

    for dx, dy in offsets:
        x = max(2, min(w-2, x0 + dx))
        y = max(12, min(h-2, y0 + dy))

        taken = False
        for (dxp, dyp) in drawn_positions:
            if (x - dxp)**2 + (y - dyp)**2 <= radius**2:
                taken = True
                break
        if not taken:
            chosen = (x, y)
            break

    if chosen is None:
        x = max(2, min(w-2, x0))
        y = max(12, min(h-2, y0))
        chosen = (x, y)

    x, y = chosen
    ((text_w, text_h), baseline) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    marker_w = max(5, int(text_h * 0.85))
    marker_h = marker_w
    spacing = 3

    if not outline:
        cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])), thickness, cv2.LINE_AA)
    else:
        marker_x1 = max(2, x - marker_w - spacing)
        marker_y1 = max(2, y - text_h)
        marker_x2 = min(w-2, marker_x1 + marker_w)
        marker_y2 = min(h-2, marker_y1 + marker_h)
        cv2.rectangle(img_bgr, (marker_x1, marker_y1), (marker_x2, marker_y2),
                      (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])), -1)
        text_x = x
        if marker_x2 + spacing + 2 > x:
            text_x = marker_x2 + spacing
        cv2.putText(img_bgr, text, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    drawn_positions.append((x, y))

def _match_mask_to_nearest_box(cx, cy, boxes_arr):
    if boxes_arr is None or len(boxes_arr) == 0:
        return None
    try:
        box_centers = np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in boxes_arr])
        dists = np.hypot(box_centers[:,0] - cx, box_centers[:,1] - cy)
        idx = int(np.argmin(dists))
        return idx
    except Exception:
        return None

# ---------------------------
# Core detect_engine (kept unchanged)
# ---------------------------
def detect_engine(image, micron, conf_thr, iou_thr, analysis_type, sub_mode):
    if isinstance(analysis_type, str) and "Qualitative" in analysis_type:
        analysis_key = "Qualitative"
    else:
        analysis_key = "Quantitative"

    if analysis_key == "Qualitative":
        model_path = QUAL_SINGLE_PT if sub_mode == "Single" else QUAL_MULTI_PT
        task = "detect"; prefer_polygons = False
    else:
        model_path = QUANT_SINGLE_PT if sub_mode == "Single" else QUANT_MULTI_PT
        task = "segment"; prefer_polygons = True

    if model_path is None:
        raise FileNotFoundError("Model path for chosen slot not found.")

    model = load_model(model_path)
    try:
        results = model(image, conf=conf_thr, iou=iou_thr, task=task, max_det=1200)
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {e}")
    r = results[0]

    annotated_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    boxes_arr = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else np.array([])
    confs_arr = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else np.array([])
    clsids_arr = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, "cls") else np.array([])

    model_names = getattr(model, "names", None)
    class_list = []
    if isinstance(model_names, dict):
        sorted_items = sorted(model_names.items(), key=lambda x: x[0])
        class_list = [v for k,v in sorted_items]
    elif isinstance(model_names, (list,tuple)):
        class_list = list(model_names)
    else:
        class_list = []

    class_color_map = {}
    palette_idx = 0
    for cls_name in class_list:
        up = cls_name.upper()
        if up in ("IP","NP","RP"):
            class_color_map[cls_name] = COLOR_MULTI_MAP[up]
        else:
            mapped = None
            ln = cls_name.lower()
            if "network" in ln or "net" in ln or "web" in ln:
                mapped = "NP"
            elif "round" in ln or "circular" in ln or ln.startswith("r"):
                mapped = "RP"
            elif "irregular" in ln or "irreg" in ln or ln.startswith("i"):
                mapped = "IP"
            if mapped:
                class_color_map[cls_name] = COLOR_MULTI_MAP[mapped]
            else:
                class_color_map[cls_name] = PALETTE[palette_idx % len(PALETTE)]
                palette_idx += 1
    if not class_list:
        class_list = ["Pore"]; class_color_map["Pore"] = PALETTE[0]

    rows = []
    mask_used = hasattr(r, "masks") and r.masks is not None
    drawn_label_positions = []

    if prefer_polygons and mask_used:
        masks_xy = getattr(r.masks, "xy", [])
        for i_mask, seg in enumerate(masks_xy):
            pts = seg_to_pts(seg)
            if pts is None or len(pts) < 3:
                continue
            poly_area = polygon_area_shoelace(pts)
            if poly_area <= 0:
                continue
            perim_px = polygon_perimeter(pts)
            cx, cy = polygon_centroid(pts)
            minx, miny, wbox, hbox = polygon_bbox(pts)
            convex_area = convex_hull_area(pts)
            circularity = (4.0 * math.pi * poly_area / (perim_px**2)) if perim_px > 0 else None
            solidity = (poly_area / convex_area) if (convex_area and convex_area > 0) else None

            conf = _get_conf_for_index(r, i_mask)

            clsid = None
            try:
                if isinstance(clsids_arr, np.ndarray) and len(clsids_arr) > i_mask:
                    clsid = int(clsids_arr[i_mask])
            except Exception:
                clsid = None

            if clsid is None:
                tmp = getattr(r.boxes, "cls", None)
                tmp_arr = None
                try:
                    if tmp is not None:
                        tmp_arr = tmp.cpu().numpy() if hasattr(tmp, "cpu") else np.array(tmp)
                except Exception:
                    try:
                        tmp_arr = np.array(list(tmp))
                    except Exception:
                        tmp_arr = None
                if tmp_arr is not None and len(tmp_arr) > i_mask:
                    try:
                        clsid = int(tmp_arr[i_mask])
                    except Exception:
                        clsid = None

            if clsid is None:
                try:
                    if cx is not None and cy is not None:
                        matched_idx = _match_mask_to_nearest_box(cx, cy, boxes_arr)
                        if matched_idx is not None and len(clsids_arr) > matched_idx:
                            clsid = int(clsids_arr[matched_idx])
                            if len(confs_arr) > matched_idx:
                                conf = float(confs_arr[matched_idx])
                except Exception:
                    pass

            if clsid is None:
                clsid = 0

            if isinstance(model_names, dict):
                cls_name = model_names.get(clsid, str(clsid))
            elif isinstance(model_names, (list,tuple)):
                cls_name = model_names[clsid] if clsid < len(model_names) else str(clsid)
            else:
                cls_name = "Pore"

            if sub_mode == "Single":
                color = COLOR_SINGLE
            else:
                color = class_color_map.get(cls_name, PALETTE[0])

            contour = pts.reshape(-1,1,2).astype(np.int32)
            cv2.polylines(annotated_bgr, [contour], True, color, 2)

            if sub_mode == "Multi":
                short = short_label(cls_name)
                label_txt = f"{short}{conf:.2f}"
            else:
                label_txt = f"{conf:.2f}"

            try:
                label_x = int(minx) if minx is not None else (int(cx) if cx is not None else 2)
                label_y = int(miny) - 8 if (miny is not None) else (int(cy) - 8 if cy is not None else 14)
            except Exception:
                label_x, label_y = (int(cx) if cx is not None else 2, int(cy) if cy is not None else 14)

            h_img, w_img = annotated_bgr.shape[:2]
            label_x = max(2, min(w_img - 4, label_x))
            label_y = max(14, min(h_img - 2, label_y))
            text_pos = (label_x, label_y)

            draw_label_once(annotated_bgr, label_txt, text_pos, color, drawn_label_positions, radius=6, font_scale=0.45, thickness=1, outline=False)

            pts_int = pts.astype(int)
            poly_str = ";".join([f"{int(x)},{int(y)}" for x,y in pts_int])
            poly_area_um2 = round(poly_area * (micron ** 2), 6) if micron else None
            perim_um = round(perim_px * micron, 6) if micron else None

            rows.append({
                "Class": cls_name,
                "Confidence": round(conf,3),
                "X": round(float(cx),3) if cx is not None else None,
                "Y": round(float(cy),3) if cy is not None else None,
                "Width (px)": round(float(wbox),3) if wbox is not None else None,
                "Height (px)": round(float(hbox),3) if hbox is not None else None,
                "Area (px²)": int(round(poly_area)),
                "Area (µm²)": round(poly_area_um2, 6) if poly_area_um2 is not None else None,
                "Polygon": poly_str,
                "Poly_Area_px": round(poly_area, 3),
                "Poly_Area_um2": round(poly_area_um2, 6) if poly_area_um2 is not None else None,
                "Perimeter_px": round(perim_px, 3),
                "Perimeter_um": round(perim_um, 6) if perim_um is not None else None,
                "Centroid_X": round(float(cx), 6) if cx is not None else None,
                "Centroid_Y": round(float(cy), 6) if cy is not None else None,
                "BBox_minX": round(float(minx), 6) if minx is not None else None,
                "BBox_minY": round(float(miny), 6) if miny is not None else None,
                "BBox_width": round(float(wbox), 6) if wbox is not None else None,
                "BBox_height": round(float(hbox), 6) if hbox is not None else None,
                "Circularity": round(float(circularity), 6) if circularity is not None else None,
                "ConvexHullArea_px": round(float(convex_area), 3),
                "Solidity": round(float(solidity), 6) if solidity is not None else None
            })

    else:
        for i, box in enumerate(boxes_arr):
            x1, y1, x2, y2 = map(int, box[:4])
            w = int(x2 - x1); h = int(y2 - y1)
            area_px = w * h
            if area_px <= 0:
                continue

            conf = _get_conf_for_index(r, i)

            clsid = int(clsids_arr[i]) if i < len(clsids_arr) else 0
            if isinstance(model_names, dict):
                cls_name = model_names.get(clsid, str(clsid))
            elif isinstance(model_names, (list,tuple)):
                cls_name = model_names[clsid] if clsid < len(model_names) else str(clsid)
            else:
                cls_name = "Pore"

            if sub_mode == "Single":
                color = COLOR_SINGLE
            else:
                color = class_color_map.get(cls_name, PALETTE[0])

            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)

            if sub_mode == "Multi":
                short = short_label(cls_name)
                label_txt = f"{short}{conf:.2f}"
            else:
                label_txt = f"{conf:.2f}"

            text_x = max(2, x1 + 2)
            text_y = max(14, y1 - 6)
            text_pos = (text_x, text_y)

            draw_label_once(annotated_bgr, label_txt, text_pos, color, drawn_label_positions, radius=14, font_scale=0.45, thickness=1, outline=False)

            poly_str = f"{x1},{y1};{x2},{y1};{x2},{y2};{x1},{y2}"
            area_um2 = round(area_px * (micron ** 2), 6) if micron else None

            rows.append({
                "Class": cls_name,
                "Confidence": round(conf,3),
                "X": int(x1),
                "Y": int(y1),
                "Width (px)": int(w),
                "Height (px)": int(h),
                "Area (px²)": int(area_px),
                "Area (µm²)": round(area_um2, 6) if area_um2 is not None else None,
                "Polygon": poly_str
            })

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    if len(rows) == 0:
        if analysis_key == "Quantitative":
            df = pd.DataFrame(columns=["Class","Confidence","X","Y","Width (px)","Height (px)","Area (px²)","Area (µm²)","Polygon",
                                       "Poly_Area_px","Poly_Area_um2","Perimeter_px","Perimeter_um","Centroid_X","Centroid_Y",
                                       "BBox_minX","BBox_minY","BBox_width","BBox_height","Circularity","ConvexHullArea_px","Solidity"])
        else:
            df = pd.DataFrame(columns=["Class","Confidence","X","Y","Width (px)","Height (px)","Area (px²)"])
        excel_path = None
        hist_path = None
        count_viz_path = None
    else:
        df_all = pd.DataFrame(rows)
        if analysis_key == "Quantitative":
            cols = ["Class","Confidence","X","Y","Width (px)","Height (px)","Area (px²)","Area (µm²)",
                    "Polygon","Poly_Area_px","Poly_Area_um2","Perimeter_px","Perimeter_um",
                    "Centroid_X","Centroid_Y","BBox_minX","BBox_minY","BBox_width","BBox_height",
                    "Circularity","ConvexHullArea_px","Solidity"]
            df = df_all.reindex(columns=[c for c in cols if c in df_all.columns])
        else:
            df = df_all[["Class","Confidence","X","Y","Width (px)","Height (px)","Area (px²)"]].copy()

        excel_path = dataframe_to_excel_bytes(df)

        if sub_mode == "Single":
            hist_path = hist_single(df["Area (px²)"].tolist())
            count_viz_path = None
        else:
            area_dict = {"IP": [], "NP": [], "RP": []}
            mapped_counts = {"IP": 0, "NP": 0, "RP": 0}
            
            for _, row in df_all.iterrows():
                cn = str(row.get("Class","")); up = cn.upper(); mapped = None
                if up in ("IP","NP","RP"):
                    mapped = up
                else:
                    cnlow = cn.lower()
                    if "network" in cnlow or "net" in cnlow:
                        mapped = "NP"
                    elif "round" in cnlow or "circular" in cnlow or cnlow.startswith("r"):
                        mapped = "RP"
                    elif "irregular" in cnlow or "irreg" in cnlow or cnlow.startswith("i"):
                        mapped = "IP"
                if mapped is None:
                    mapped = "IP"
                area_dict[mapped].append(float(row.get("Area (px²)", 0)))
                mapped_counts[mapped] += 1
            
            hist_path = bar_chart_area(area_dict)
            count_viz_path = create_count_visualization(mapped_counts)

    if df.empty:
        summary = "No detections."
    else:
        if sub_mode == "Single":
            summary = f"Total detections: {len(df)}"
        else:
            mapped_counts = {"IP": 0, "NP": 0, "RP": 0}
            for cn in df["Class"].astype(str).tolist():
                up = cn.upper()
                mapped = None
                if up in ("IP", "NP", "RP"):
                    mapped = up
                else:
                    ln = cn.lower()
                    if "network" in ln or "net" in ln or "web" in ln:
                        mapped = "NP"
                    elif "round" in ln or "circular" in ln or ln.startswith("r"):
                        mapped = "RP"
                    elif "irregular" in ln or "irreg" in ln or ln.startswith("i"):
                        mapped = "IP"
                if mapped is None:
                    mapped = "IP"
                mapped_counts[mapped] += 1

            # Use renamed labels in summary
            summary = f"{LABEL_MAP['IP']} = {mapped_counts['IP']}, {LABEL_MAP['NP']} = {mapped_counts['NP']}, {LABEL_MAP['RP']} = {mapped_counts['RP']}"

    # Return count visualization path
    if sub_mode == "Multi":
        return annotated_rgb, df, excel_path, hist_path, summary, count_viz_path
    else:
        return annotated_rgb, df, excel_path, hist_path, summary, None

# ---------------------------
# compute metrics helper
# ---------------------------
def compute_overall_metrics(df, image_shape, micron):
    if df is None or df.empty:
        return {"count":0, "total_area_px":0.0, "area_fraction_pct":0.0, "total_area_um2":0.0}
    if "Area (px²)" in df.columns:
        areas_px = df["Area (px²)"].astype(float).tolist()
    elif "Poly_Area_px" in df.columns:
        areas_px = df["Poly_Area_px"].astype(float).tolist()
    else:
        if "Width (px)" in df.columns and "Height (px)" in df.columns:
            areas_px = (df["Width (px)"].astype(float) * df["Height (px)"].astype(float)).tolist()
        else:
            areas_px = []
    total_area_px = float(np.sum(areas_px)) if len(areas_px)>0 else 0.0
    H, W = image_shape[0], image_shape[1]
    image_area_px = float(H * W) if H and W else 1.0
    area_fraction = 100.0 * (total_area_px / image_area_px) if image_area_px>0 else 0.0
    total_area_um2 = total_area_px * (micron ** 2) if micron else None
    return {"count": len(areas_px), "total_area_px": total_area_px, "area_fraction_pct": area_fraction, "total_area_um2": total_area_um2}

# ---------------------------
# GRADIO UI (center image + centered controls + summary cards)
# ---------------------------
CUSTOM_CSS = """
/* Lightened header & full-width description card with orange header */
body { font-family: Inter, Arial, sans-serif; }
.gradio-container { background: #fbfdff; padding: 12px; }

/* Header: orange gradient now (matches Analyze) */
.header {
  background: linear-gradient(90deg,#ff7a18,#ffb347); /* orange gradient */
  color:white;
  padding:14px 18px;
  border-radius:8px;
  box-shadow: 0 6px 18px rgba(8,20,40,0.06);
  width: 100%;
  box-sizing: border-box;
  margin-bottom: 12px;
}

/* Title style */
.header h3 { margin:0; font-weight:700; font-size:1.05rem; color: #ffffff; }

/* Description card spans full width and is readable */
.desc-card {
  background: #fff8f2;
  border: 1px solid #ffe6d0;
  padding: 12px 16px;
  border-radius: 6px;
  color: #3a2b1f;
  box-shadow: 0 4px 14px rgba(8,20,40,0.04);
  margin-bottom: 14px;
  width: 100%;
  box-sizing: border-box;
  font-size: 0.95rem;
  line-height: 1.35;
}

/* existing summary card */
.summary-card {
  border: 1px solid #e6e9ef;
  background: #fffaf6;
  padding: 8px 12px;
  border-radius: 6px;
  font-weight: 600;
  color: #1f3b5f;
  display: inline-block;
}

/* center image sizing */
.center-block { display:flex; justify-content:center; align-items:center; }
#input-image img { max-width: 720px; width: 60vw; height: auto; }

/* center controls */
.controls-center { display:flex; justify-content:center; align-items:center; }
.controls-inner { width: 85%; }

/* small helper */
.small-muted { color:#444; font-size:0.9rem; }
"""

# exact description text provided by user
description_markdown = """
Description: Detect and classify discontinuities in your microstructure using this tool, which extends a regression-based detection algorithm and is available in two variants: Qualitative (Number-Based) and Quantitative (Area-Based). The Qualitative mode detects all discontinuities, categorizes them, and provides the total pore count, with area estimations calculated using bounding boxes. The Quantitative mode classifies porosities into Irregular, Round, and Network types based on morphology, using polygons to compute the exact area fraction of each discontinuity. In the Porosity Analysis Table, the areas derived from bounding boxes or polygon measurements are summarized, and their distribution is visualized as a histogram.
"""

with gr.Blocks(css=CUSTOM_CSS, title="QPoroLyser — Centered UI") as demo:
    # Full-width header row
    with gr.Row():
        # single full-width column for header
        with gr.Column(scale=1):
            gr.HTML("<div class='header'><h3>QPoroLyser</h3></div>")

    # Full-width description card (single row, single column)
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(f"<div class='desc-card'>{description_markdown}</div>")

    # Centered Input image row (absolute center)
    with gr.Row():
        with gr.Column(scale=0.15):
            pass
        with gr.Column(scale=0.7):
            with gr.Row(elem_id="center-image-row"):
                with gr.Column():
                    inp = gr.Image(type="numpy", label="Input Microstructure", elem_id="input-image")
        with gr.Column(scale=0.15):
            pass

    # Controls row centered beneath image
    with gr.Row():
        with gr.Column(scale=0.15):
            pass
        with gr.Column(scale=0.7):
            # controls wrapped for compatibility
            with gr.Row():
                with gr.Column():
                    with gr.Row(elem_id="controls-center", variant="compact"):
                        with gr.Column(scale=1, elem_classes="controls-center"):
                            with gr.Row(elem_id="controls-inner"):
                                micron = gr.Number(label="µm per pixel", value=0.1, info="Physical size per pixel. Use for µm² outputs.")
                                conf_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Confidence Threshold")
                                iou_slider  = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="IOU Threshold")
                            analysis_type = gr.Radio(
                                choices=["Qualitative Analysis(Number Based)", "Quantitative Analysis.(Area Based)"],
                                value="Qualitative Analysis(Number Based)",
                                label="Select Analysis Type"
                            )
                            analyze_btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=0.15):
            pass

    # Results row: two columns (Single / Multi). Each column shows a summary card above the annotated image.
    with gr.Row():
        # Single-class column
        with gr.Column(scale=0.5):
            gr.HTML("<div style='font-weight:700; margin-bottom:6px'>Single-class Results</div>")
            overall_single_summary = gr.HTML("<div class='summary-card'>Single-class metrics will appear here</div>")
            out_img_single = gr.Image(type="numpy", label="Annotated (Single-class)")
            out_table_single = gr.Dataframe(label="Single-class Table", wrap=True)
            out_excel_single = gr.File(label="Download Single Excel")
            out_hist_single = gr.Image(type="filepath", label="Single Histogram")
            out_summary_single = gr.Textbox(label="Single Summary", interactive=False)

        # Multi-class column
        with gr.Column(scale=0.5):
            gr.HTML("<div style='font-weight:700; margin-bottom:6px'>Multi-class Results</div>")
            overall_multi_summary = gr.HTML("<div class='summary-card'>Multi-class metrics will appear here</div>")
            out_img_multi = gr.Image(type="numpy", label="Annotated (Multi-class)")
            out_table_multi = gr.Dataframe(label="Multi-class Table", wrap=True)
            out_excel_multi = gr.File(label="Download Multi Excel")
            out_hist_multi = gr.Image(type="filepath", label="Multi Histogram (Area)")
            out_count_viz_multi = gr.Image(type="filepath", label="Defect Fingerprint (Count)")
            out_summary_multi = gr.Textbox(label="Multi Summary", interactive=False)

    gr.Markdown("**Note:** Both Single and Multi analyses are executed automatically for the chosen mode.")

    # Analyze wrapper
    def analyze_click(img, micron_val, conf_val, iou_val, analysis_val):
        if img is None:
            empty_df = pd.DataFrame([])
            msg = "Upload an image first."
            empty_card = "<div class='summary-card'>Upload an image to see metrics</div>"
            return (None, empty_df, None, None, msg, None, empty_df, None, None, None, msg, empty_card, empty_card)

        analysis_key = "Qualitative" if (isinstance(analysis_val, str) and "Qualitative" in analysis_val) else "Quantitative"

        # run Single
        try:
            ann_s, df_s, excel_s, hist_s, summary_s, _ = detect_engine(img, micron_val, conf_val, iou_val, analysis_key, "Single")
        except Exception as e:
            ann_s, df_s, excel_s, hist_s, summary_s = None, pd.DataFrame([]), None, None, f"Error (Single): {e}"

        # run Multi
        try:
            ann_m, df_m, excel_m, hist_m, summary_m, count_viz_m = detect_engine(img, micron_val, conf_val, iou_val, analysis_key, "Multi")
        except Exception as e:
            ann_m, df_m, excel_m, hist_m, summary_m, count_viz_m = None, pd.DataFrame([]), None, None, f"Error (Multi): {e}", None

        # compute overall metrics
        metrics_s = compute_overall_metrics(df_s, img.shape, micron_val)
        metrics_m = compute_overall_metrics(df_m, img.shape, micron_val)

        def pretty_metrics_card(m):
            if m["count"] == 0:
                return "<div class='summary-card'>No detections.</div>"
            area_um2 = m["total_area_um2"]
            if area_um2 is None:
                area_um2_str = "N/A"
            else:
                area_um2_str = f"{area_um2:.3f} µm²"
            return "<div class='summary-card'>Count: {count} &nbsp; · &nbsp; Total area: {area_px:.1f} px² ({area_um2}) <br/> Area fraction: {frac:.3f}%</div>".format(
                count=m["count"], area_px=m["total_area_px"], area_um2=area_um2_str, frac=m["area_fraction_pct"]
            )

        card_s = pretty_metrics_card(metrics_s)
        card_m = pretty_metrics_card(metrics_m)

        return (ann_s, df_s, excel_s, hist_s, summary_s, ann_m, df_m, excel_m, hist_m, count_viz_m, summary_m, card_s, card_m)

    analyze_btn.click(
        fn=analyze_click,
        inputs=[inp, micron, conf_slider, iou_slider, analysis_type],
        outputs=[out_img_single, out_table_single, out_excel_single, out_hist_single, out_summary_single,
                 out_img_multi, out_table_multi, out_excel_multi, out_hist_multi, out_count_viz_multi, out_summary_multi,
                 overall_single_summary, overall_multi_summary]
    )

if __name__ == "__main__":
    demo.launch()
import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Map folder/filename group codes to experimental design labels
GROUP_MAP = {
    "CC": ("Cotton", "Control", "Control"),
    "CP": ("Polyester", "Control", "Control"),
    "CI": ("Cotton", "Indoor", "Indoor"),
    "PI": ("Polyester", "Indoor", "Indoor"),
    "C1": ("Cotton", "Outdoor", "Site 1"),
    "C2": ("Cotton", "Outdoor", "Site 2"),
    "C3": ("Cotton", "Outdoor", "Site 3"),
    "P1": ("Polyester", "Outdoor", "Site 1"),
    "P2": ("Polyester", "Outdoor", "Site 2"),
    "P3": ("Polyester", "Outdoor", "Site 3"),
}

# Matches your renamed filenames: e.g., C1_D2_1.jpg, PI_D14_8.jpg, CC_D0_1.jpg
FILENAME_PATTERN = re.compile(
    r"^(CC|CP|CI|PI|C[1-3]|P[1-3])_D(\d+)_(\d+)$",
    re.IGNORECASE
)

# Candidate ROI dimensions (width, height) allowing variable high-focus data capture
CANDIDATE_ROI_SIZES = [
    (850, 650),
    (750, 600),
    (650, 600),
    (600, 500),
    (500, 500),
]

# Calibrated thresholds
MIN_FOCUS_THRESHOLD = 5.0
MAX_CLEAN_DELTA_E = 5.5      # Max color shift from Day 0 control to still be considered clean (Grade A)
MAX_CLEAN_YELLOW_SHIFT = 3.5 # Max yellowing shift (+b*) to still be considered clean (Grade A)
MAX_CLEAN_SOIL_PCT = 1.5     # Max % of dark/yellowish soil pixels inside ROI for Grade A


def bgr_to_true_cielab(roi_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts OpenCV 8-bit BGR image to standard perceptual CIELAB (L*: 0..100, a*, b*: -128..127)."""
    lab_cv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_star = lab_cv[:, :, 0] * (100.0 / 255.0)
    a_star = lab_cv[:, :, 1] - 128.0
    b_star = lab_cv[:, :, 2] - 128.0
    return L_star, a_star, b_star


def find_best_focus_roi(img_bgr: np.ndarray, margin_x: int = 140, margin_y: int = 90, stride: int = 40) -> dict:
    """
    Searches across variable window sizes and positions inside the vignette-free
    interior of an image to locate the sharpest, highest-data Region of Interest (ROI).
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_sq = lap ** 2

    max_area = max(cw * ch for cw, ch in CANDIDATE_ROI_SIZES)
    best_result = {
        "roi_x": max(0, (w - 600) // 2),
        "roi_y": max(0, (h - 600) // 2),
        "roi_w": min(w, 600),
        "roi_h": min(h, 600),
        "focus_score": 0.0,
        "combined_score": -1.0,
    }

    x_min_bound, x_max_bound = margin_x, w - margin_x
    y_min_bound, y_max_bound = margin_y, h - margin_y

    for cw, ch in CANDIDATE_ROI_SIZES:
        if (x_max_bound - x_min_bound) < cw or (y_max_bound - y_min_bound) < ch:
            continue

        area_weight = ((cw * ch) / max_area) ** 0.30

        for y in range(y_min_bound, y_max_bound - ch + 1, stride):
            for x in range(x_min_bound, x_max_bound - cw + 1, stride):
                patch_gray = gray[y:y + ch, x:x + cw]
                if patch_gray.mean() < 45.0:
                    continue

                mean_sq = float(lap_sq[y:y + ch, x:x + cw].mean())
                mean_val = float(lap[y:y + ch, x:x + cw].mean())
                focus_var = max(0.0, mean_sq - (mean_val ** 2))
                combined_score = focus_var * area_weight

                if combined_score > best_result["combined_score"]:
                    best_result = {
                        "roi_x": int(x),
                        "roi_y": int(y),
                        "roi_w": int(cw),
                        "roi_h": int(ch),
                        "focus_score": round(focus_var, 2),
                        "combined_score": combined_score,
                    }

    return best_result


def audit_dataset(raw_dir: Path, min_focus: float = MIN_FOCUS_THRESHOLD) -> pd.DataFrame:
    """
    Two-pass dataset audit:
      Pass 1: Scans all images, maps sampling sessions (D0..D15) to actual exposure
              duration in days (0, 2, 4, ..., 30), finds optimal high-focus ROI,
              and extracts mean CIELAB.
      Pass 2: Measures visual soil/stain deviation against Day 0 controls (CC/CP)
              to objectively separate clean Grade A samples from soil-stained Grade B samples.
    """
    records = []
    roi_lab_cache = {}
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    # --- PASS 1: Metadata parsing, adaptive ROI detection & raw CIELAB measurement ---
    for img_path in sorted(raw_dir.rglob("*")):
        if img_path.suffix.lower() not in valid_exts:
            continue

        match = FILENAME_PATTERN.match(img_path.stem)
        if match:
            group_code, session_str, sample_str = match.groups()
            group_code = group_code.upper()
            fabric, setting, site = GROUP_MAP.get(group_code, ("Unknown", "Unknown", "Unknown"))
            sampling_session = int(session_str)      # D0..D15
            day = sampling_session * 2               # Actual exposure duration: 0, 2, 4, ..., 30 days
            sample_num = int(sample_str)
        else:
            group_code = img_path.parent.name.upper()
            fabric, setting, site = GROUP_MAP.get(group_code, ("Unknown", "Unknown", "Unknown"))
            sampling_session, day, sample_num = -1, -1, -1

        img = cv2.imread(str(img_path))
        if img is None:
            records.append({
                "image_id": img_path.stem, "group_code": group_code, "fabric": fabric,
                "setting": setting, "site": site,
                "sampling_session": sampling_session, "day": day, "sample_num": sample_num,
                "rel_path": str(img_path.relative_to(raw_dir)),
                "width_px": 0, "height_px": 0,
                "roi_x": 0, "roi_y": 0, "roi_w": 0, "roi_h": 0,
                "focus_score": 0.0, "raw_L": 0.0, "raw_a": 0.0, "raw_b": 0.0,
                "raw_delta_e": 0.0, "yellow_shift": 0.0, "soil_pixel_pct": 0.0,
                "qc_grade": "C", "qc_reason": "corrupt_file", "read_status": "corrupt", "eligible": False
            })
            continue

        h, w = img.shape[:2]
        roi_info = find_best_focus_roi(img)
        rx, ry, rw, rh = roi_info["roi_x"], roi_info["roi_y"], roi_info["roi_w"], roi_info["roi_h"]

        roi_bgr = img[ry:ry + rh, rx:rx + rw]
        L_star, a_star, b_star = bgr_to_true_cielab(roi_bgr)
        roi_lab_cache[img_path.stem] = (L_star, a_star, b_star)

        records.append({
            "image_id": img_path.stem,
            "group_code": group_code,
            "fabric": fabric,
            "setting": setting,
            "site": site,
            "sampling_session": sampling_session,
            "day": day,
            "sample_num": sample_num,
            "rel_path": str(img_path.relative_to(raw_dir)),
            "width_px": w,
            "height_px": h,
            "roi_x": rx,
            "roi_y": ry,
            "roi_w": rw,
            "roi_h": rh,
            "focus_score": roi_info["focus_score"],
            "raw_L": round(float(L_star.mean()), 2),
            "raw_a": round(float(a_star.mean()), 2),
            "raw_b": round(float(b_star.mean()), 2),
            "read_status": "ok",
        })

    df = pd.DataFrame(records)

    # Compute clean Day 0 baseline CIELAB per fabric type (CC for Cotton, CP for Polyester)
    baselines = {}
    for fab in ["Cotton", "Polyester"]:
        ctrl_sub = df[(df["fabric"] == fab) & (df["setting"] == "Control") & (df["read_status"] == "ok")]
        if len(ctrl_sub) > 0:
            baselines[fab] = (
                float(ctrl_sub["raw_L"].mean()),
                float(ctrl_sub["raw_a"].mean()),
                float(ctrl_sub["raw_b"].mean()),
            )
        else:
            baselines[fab] = (82.0, 0.0, -3.0)

    # --- PASS 2: Visual soil/stain evaluation & final QC Grade assignment ---
    delta_es, yellow_shifts, soil_pcts, grades, reasons, eligibles = [], [], [], [], [], []

    for _, row in df.iterrows():
        if row["read_status"] != "ok":
            delta_es.append(0.0)
            yellow_shifts.append(0.0)
            soil_pcts.append(0.0)
            grades.append("C")
            reasons.append("corrupt_file")
            eligibles.append(False)
            continue

        L0, a0, b0 = baselines.get(row["fabric"], (82.0, 0.0, -3.0))
        L_star, a_star, b_star = roi_lab_cache[row["image_id"]]

        de = float(np.sqrt((row["raw_L"] - L0) ** 2 + (row["raw_a"] - a0) ** 2 + (row["raw_b"] - b0) ** 2))
        y_shift = float(row["raw_b"] - b0)
        soil_mask = (L_star < (L0 - 12.0)) | (b_star > (b0 + 6.0))
        soil_pct = float(soil_mask.mean() * 100.0)

        delta_es.append(round(de, 2))
        yellow_shifts.append(round(y_shift, 2))
        soil_pcts.append(round(soil_pct, 2))

        if row["width_px"] < 850 or row["height_px"] < 650:
            grade, reason = "C", f"resolution_too_small ({row['width_px']}x{row['height_px']})"
        elif row["focus_score"] < min_focus:
            grade, reason = "C", f"blurry_focus ({row['focus_score']} < {min_focus})"
        elif row["setting"] in {"Control", "Indoor"}:
            grade, reason = "A", "clean_optimal"
        else:
            is_dirty = (
                (de >= MAX_CLEAN_DELTA_E)
                or (y_shift >= MAX_CLEAN_YELLOW_SHIFT)
                or (soil_pct >= MAX_CLEAN_SOIL_PCT)
            )
            if is_dirty:
                grade, reason = "B", "outdoor_soil_stained"
            else:
                grade, reason = "A", "outdoor_clean_optimal"

        grades.append(grade)
        reasons.append(reason)
        eligibles.append(grade in {"A", "B"})

    df["raw_delta_e"] = delta_es
    df["yellow_shift"] = yellow_shifts
    df["soil_pixel_pct"] = soil_pcts
    df["qc_grade"] = grades
    df["qc_reason"] = reasons
    df["eligible"] = eligibles

    return df

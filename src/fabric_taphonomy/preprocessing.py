from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def extract_adaptive_roi(img_bgr: np.ndarray, roi_x: int, roi_y: int, roi_w: int, roi_h: int) -> np.ndarray:
    """
    Crops the image using the custom high-focus Region of Interest (ROI) coordinates
    determined in Notebook 01, removing outer microscope vignetting and out-of-focus regions.
    """
    return img_bgr[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].copy()


def prepare_texture_grayscale(roi_bgr: np.ndarray, qc_grade: str, median_ksize: int = 3) -> np.ndarray:
    """
    Converts the cropped color ROI into an 8-bit grayscale image for texture analysis.
    - Grade A (clean): Converted directly to grayscale without filtering to preserve fine fiber micro-detail.
    - Grade B (soil-stained): Applies a 1-px radius (3x3 kernel) median filter to suppress discrete
      surface soil particles while preserving sharp structural thread weave edges.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    if qc_grade == "B":
        return cv2.medianBlur(gray, median_ksize)
    return gray


def run_batch_preprocessing(
    inventory_df: pd.DataFrame,
    raw_dir: Path,
    cropped_dir: Path,
    texture_dir: Path,
    median_ksize: int = 3,
) -> pd.DataFrame:
    """
    Processes all eligible (Grade A and Grade B) images:
      1. Extracts the per-image adaptive high-focus ROI and saves it as a lossless .png
         in data/processed/cropped_roi/ (for CIELAB & HSV color extraction).
      2. Generates the grayscale texture-ready image (applying 3x3 median filtering only to
         Grade B images) and saves it as a lossless .png in data/processed/filtered_texture/.
    Returns an updated manifest DataFrame with exact paths to processed files.
    """
    eligible_df = inventory_df[inventory_df["eligible"]].copy().reset_index(drop=True)
    cropped_rel_paths = []
    texture_rel_paths = []

    for _, row in eligible_df.iterrows():
        src_path = raw_dir / row["rel_path"]
        img_bgr = cv2.imread(str(src_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read eligible raw image: {src_path}")

        # 1. Adaptive high-focus ROI crop (Color Branch)
        roi_bgr = extract_adaptive_roi(
            img_bgr,
            roi_x=int(row["roi_x"]),
            roi_y=int(row["roi_y"]),
            roi_w=int(row["roi_w"]),
            roi_h=int(row["roi_h"]),
        )

        # Save as lossless .png inside group subfolder (e.g., C1/C1_D1_1.png)
        rel_png = Path(row["group_code"]) / f"{row['image_id']}.png"
        out_crop_path = cropped_dir / rel_png
        out_crop_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_crop_path), roi_bgr)

        # 2. Grayscale conversion + Grade B median filtering (Texture Branch)
        tex_gray = prepare_texture_grayscale(
            roi_bgr,
            qc_grade=str(row["qc_grade"]),
            median_ksize=median_ksize,
        )

        out_tex_path = texture_dir / rel_png
        out_tex_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_tex_path), tex_gray)

        cropped_rel_paths.append(str(rel_png))
        texture_rel_paths.append(str(rel_png))

    eligible_df["cropped_rel_path"] = cropped_rel_paths
    eligible_df["texture_rel_path"] = texture_rel_paths
    return eligible_df
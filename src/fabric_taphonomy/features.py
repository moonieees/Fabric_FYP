from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def extract_color_features(roi_bgr: np.ndarray) -> dict:
    """
    Extracts perceptual color features from an unfiltered adaptive ROI image:
      1. CIELAB (L*: Lightness 0..100, a*: Green-Red -128..127, b*: Blue-Yellow -128..127)
         including both mean values and spatial standard deviations (surface heterogeneity).
      2. HSV (Circular Mean Hue 0..360 deg, Saturation 0..100%, Value 0..100%).
    """
    # --- 1. Perceptual CIELAB Color Space ---
    lab_cv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_star = lab_cv[:, :, 0] * (100.0 / 255.0)
    a_star = lab_cv[:, :, 1] - 128.0
    b_star = lab_cv[:, :, 2] - 128.0

    # --- 2. HSV Color Space ---
    hsv_cv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_deg = hsv_cv[:, :, 0] * 2.0               # Scale OpenCV 0..179 to 0..360 degrees
    sat_pct = (hsv_cv[:, :, 1] / 255.0) * 100.0   # Scale 0..255 to 0..100%
    val_pct = (hsv_cv[:, :, 2] / 255.0) * 100.0   # Scale 0..255 to 0..100%

    # Use circular mean for Hue (0-360 deg) to prevent wrap-around distortion near red/yellow
    hue_rad = np.deg2rad(hue_deg)
    mean_sin = float(np.mean(np.sin(hue_rad)))
    mean_cos = float(np.mean(np.cos(hue_rad)))
    hue_circ_mean = float(np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360.0)

    return {
        "lab_L_mean": round(float(np.mean(L_star)), 4),
        "lab_L_std": round(float(np.std(L_star)), 4),
        "lab_a_mean": round(float(np.mean(a_star)), 4),
        "lab_a_std": round(float(np.std(a_star)), 4),
        "lab_b_mean": round(float(np.mean(b_star)), 4),
        "lab_b_std": round(float(np.std(b_star)), 4),
        "hsv_H_deg": round(hue_circ_mean, 4),
        "hsv_S_pct": round(float(np.mean(sat_pct)), 4),
        "hsv_V_pct": round(float(np.mean(val_pct)), 4),
    }


def compute_delta_e_against_controls(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes the baseline Day 0 CIELAB reference (L0*, a0*, b0*) separately for
    Cotton (group CC) and Polyester (group CP), then calculates per-channel shifts
    (ΔL*, Δa*, Δb*) and Total Color Difference (ΔE76) for every sample.
    """
    out_df = df.copy()
    baseline_rows = []

    for fab in ["Cotton", "Polyester"]:
        ctrl = out_df[(out_df["fabric"] == fab) & (out_df["setting"] == "Control")]
        if len(ctrl) == 0:
            raise ValueError(f"No Day 0 Control samples found for fabric '{fab}'!")

        L0 = float(ctrl["lab_L_mean"].mean())
        a0 = float(ctrl["lab_a_mean"].mean())
        b0 = float(ctrl["lab_b_mean"].mean())

        baseline_rows.append({
            "fabric": fab,
            "control_group": "CC" if fab == "Cotton" else "CP",
            "n_controls": len(ctrl),
            "baseline_L0": round(L0, 4),
            "baseline_a0": round(a0, 4),
            "baseline_b0": round(b0, 4),
        })

        mask = out_df["fabric"] == fab
        out_df.loc[mask, "delta_L"] = (out_df.loc[mask, "lab_L_mean"] - L0).round(4)
        out_df.loc[mask, "delta_a"] = (out_df.loc[mask, "lab_a_mean"] - a0).round(4)
        out_df.loc[mask, "delta_b"] = (out_df.loc[mask, "lab_b_mean"] - b0).round(4)
        out_df.loc[mask, "delta_E"] = np.sqrt(
            (out_df.loc[mask, "lab_L_mean"] - L0) ** 2
            + (out_df.loc[mask, "lab_a_mean"] - a0) ** 2
            + (out_df.loc[mask, "lab_b_mean"] - b0) ** 2
        ).round(4)

    baseline_df = pd.DataFrame(baseline_rows)
    return out_df, baseline_df


def extract_texture_features(gray_img: np.ndarray, canny_low: int = 25, canny_high: int = 75) -> dict:
    """
    Extracts area-normalized structural texture features from the 8-bit grayscale ROI
    (where Grade B images have already undergone 3x3 median soil-noise filtering):
      1. Rotation-Invariant GLCM (averaged across 0, 45, 90, 135 deg at distance d=1):
         Contrast, Homogeneity, Energy, and Correlation.
      2. Rotation-Invariant Uniform Local Binary Patterns (LBP, P=8, R=1):
         LBP Shannon Entropy and LBP Energy (Uniformity).
      3. Canny Edge Density (% of ROI pixels classified as structural thread/fibril edges).
    """
    # --- 1. Gray-Level Co-occurrence Matrix (GLCM) ---
    angles = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]
    glcm = graycomatrix(
        gray_img,
        distances=[1],
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    energy = float(np.mean(graycoprops(glcm, "energy")))
    correlation = float(np.mean(graycoprops(glcm, "correlation")))

    # --- 2. Uniform Local Binary Patterns (LBP) ---
    n_points = 8
    radius = 1
    lbp_map = local_binary_pattern(gray_img, P=n_points, R=radius, method="uniform")
    n_bins = n_points + 2  # 10 uniform/non-uniform bins for P=8
    hist, _ = np.histogram(lbp_map.ravel(), bins=n_bins, range=(0, n_bins), density=False)
    hist_prob = hist.astype(np.float64) / hist.sum()

    lbp_entropy = float(-np.sum(hist_prob * np.log2(hist_prob + 1e-12)))
    lbp_energy = float(np.sum(hist_prob ** 2))

    # --- 3. Canny Edge Density ---
    edges = cv2.Canny(gray_img, threshold1=canny_low, threshold2=canny_high)
    edge_density_pct = (float(np.count_nonzero(edges)) / float(edges.size)) * 100.0

    return {
        "glcm_contrast": round(contrast, 4),
        "glcm_homogeneity": round(homogeneity, 4),
        "glcm_energy": round(energy, 4),
        "glcm_correlation": round(correlation, 4),
        "lbp_entropy": round(lbp_entropy, 4),
        "lbp_energy": round(lbp_energy, 4),
        "canny_edge_pct": round(edge_density_pct, 4),
    }


def run_feature_extraction_pipeline(
    manifest_df: pd.DataFrame,
    cropped_dir: Path,
    texture_dir: Path,
    canny_low: int = 25,
    canny_high: int = 75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs full color and texture feature extraction across all preprocessed images,
    maps sampling sessions (D0..D15) to true exposure duration (0, 2, 4, ..., 30 days),
    and computes ΔE76 relative to Day 0 fabric controls.
    """
    feature_rows = []

    for _, row in manifest_df.iterrows():
        color_path = cropped_dir / row["cropped_rel_path"]
        tex_path = texture_dir / row["texture_rel_path"]

        roi_bgr = cv2.imread(str(color_path))
        tex_gray = cv2.imread(str(tex_path), cv2.IMREAD_GRAYSCALE)

        if roi_bgr is None or tex_gray is None:
            raise FileNotFoundError(f"Missing processed image for {row['image_id']}")

        color_feats = extract_color_features(roi_bgr)
        tex_feats = extract_texture_features(tex_gray, canny_low=canny_low, canny_high=canny_high)

        row_dict = row.to_dict()
        # Derive exact sampling session (D0..D15) and exposure days (0, 2, 4, ..., 30) from image_id
        session_num = int(str(row_dict["image_id"]).split("_")[1].lstrip("Dd"))
        row_dict["sampling_session"] = session_num
        row_dict["day"] = session_num * 2

        combined = {**row_dict, **color_feats, **tex_feats}
        feature_rows.append(combined)

    raw_features_df = pd.DataFrame(feature_rows)
    master_df, baseline_df = compute_delta_e_against_controls(raw_features_df)
    return master_df, baseline_df
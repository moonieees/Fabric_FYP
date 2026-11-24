from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


CORE_FEATURES = [
    "delta_E", "delta_L", "delta_b",
    "hsv_H_deg", "hsv_S_pct", "hsv_V_pct",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
    "lbp_entropy", "lbp_energy", "canny_edge_pct",
]


def assign_exposure_stage(day: int, setting: str) -> str:
    """Groups the 30-day exposure timeline into four interpretive stages for LDA comparison."""
    if setting == "Control" or day == 0:
        return "0_Control (Day 0)"
    elif day <= 10:
        return "1_Early (Days 2-10)"
    elif day <= 20:
        return "2_Mid (Days 12-20)"
    else:
        return "3_Late (Days 22-30)"


def run_two_way_anova_suite(df: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs Type-II Two-Way ANOVA (Fabric x Setting) across all selected color & texture metrics,
    followed by Tukey's HSD post-hoc pairwise comparisons across Fabric_Setting groups.
    """
    anova_rows = []
    tukey_frames = []

    work_df = df.copy()
    work_df["fabric_setting"] = work_df["fabric"] + "_" + work_df["setting"]

    for metric in metrics:
        sub = work_df[["fabric", "setting", "fabric_setting", metric]].dropna()
        model = ols(f"{metric} ~ C(fabric) + C(setting) + C(fabric):C(setting)", data=sub).fit()
        aov = sm.stats.anova_lm(model, typ=2)

        for factor in ["C(fabric)", "C(setting)", "C(fabric):C(setting)"]:
            f_val = float(aov.loc[factor, "F"])
            p_val = float(aov.loc[factor, "PR(>F)"])
            ss_effect = float(aov.loc[factor, "sum_sq"])
            ss_total = float(aov["sum_sq"].sum())
            eta_sq = ss_effect / ss_total if ss_total > 0 else 0.0

            anova_rows.append({
                "feature": metric,
                "factor": factor.replace("C(", "").replace(")", ""),
                "F_stat": round(f_val, 4),
                "p_value": round(p_val, 6),
                "eta_squared": round(eta_sq, 4),
                "significant_005": p_val < 0.05,
            })

        # Post-hoc Tukey's HSD across combined Fabric_Setting groups
        tukey = pairwise_tukeyhsd(endog=sub[metric], groups=sub["fabric_setting"], alpha=0.05)
        tukey_df = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0],
        )
        tukey_df.insert(0, "feature", metric)
        tukey_frames.append(tukey_df)

    return pd.DataFrame(anova_rows), pd.concat(tukey_frames, ignore_index=True)


def _exp_saturation(t: np.ndarray, a: float, k: float, c: float) -> np.ndarray:
    """Asymptotic exponential degradation model: y(t) = a * (1 - exp(-k * t)) + c"""
    return a * (1.0 - np.exp(-k * t)) + c


def fit_temporal_curves(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict]:
    """
    Fits non-linear exponential degradation trajectories across Exposure Days (0..30)
    for each (Fabric, Setting) combination using Day 0 controls as t=0 anchor.
    """
    fit_rows = []
    curves = {}

    for fab in ["Cotton", "Polyester"]:
        ctrl_df = df[(df["fabric"] == fab) & (df["setting"] == "Control")]

        for setting in ["Indoor", "Outdoor"]:
            exp_df = df[(df["fabric"] == fab) & (df["setting"] == setting)]
            combined = pd.concat([ctrl_df, exp_df], ignore_index=True)
            daily = combined.groupby("day")[metric].mean().reset_index().sort_values("day")

            t_data = daily["day"].to_numpy(dtype=float)
            y_data = daily[metric].to_numpy(dtype=float)

            y0 = float(y_data[0])
            y_span = float(y_data[-1] - y0)

            try:
                popt, _ = curve_fit(
                    _exp_saturation,
                    t_data,
                    y_data,
                    p0=[y_span, 0.08, y0],
                    bounds=([-200.0, 0.0001, -200.0], [200.0, 2.0, 200.0]),
                    maxfev=5000,
                )
                y_pred = _exp_saturation(t_data, *popt)
                ss_res = float(np.sum((y_data - y_pred) ** 2))
                ss_tot = float(np.sum((y_data - np.mean(y_data)) ** 2))
                r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                t_smooth = np.linspace(0, float(t_data.max()), 100)
                y_smooth = _exp_saturation(t_smooth, *popt)

                fit_rows.append({
                    "metric": metric,
                    "fabric": fab,
                    "setting": setting,
                    "amplitude_a": round(float(popt[0]), 4),
                    "rate_k_per_day": round(float(popt[1]), 4),
                    "baseline_c": round(float(popt[2]), 4),
                    "R_squared": round(r_sq, 4),
                })
                curves[(fab, setting)] = (t_data, y_data, t_smooth, y_smooth, r_sq)
            except RuntimeError:
                fit_rows.append({
                    "metric": metric,
                    "fabric": fab,
                    "setting": setting,
                    "amplitude_a": np.nan,
                    "rate_k_per_day": np.nan,
                    "baseline_c": round(y0, 4),
                    "R_squared": np.nan,
                })

    return pd.DataFrame(fit_rows), curves


def run_pca_and_lda(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Standardizes extracted color & texture features and executes:
      1. Unsupervised Principal Component Analysis (PCA)
      2. Supervised Linear Discriminant Analysis (LDA) across 30-day Exposure Stages
    """
    work_df = df.copy()
    work_df["exposure_stage"] = [
        assign_exposure_stage(int(d), str(s)) for d, s in zip(work_df["day"], work_df["setting"])
    ]

    X = work_df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. PCA
    pca = PCA(n_components=min(len(feature_cols), 5))
    pca_scores = pca.fit_transform(X_scaled)
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    ).round(4)

    # 2. LDA (supervised separation across exposure stages)
    y_stage = work_df["exposure_stage"].to_numpy()
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_scores = lda.fit_transform(X_scaled, y_stage)
    lda_accuracy = float(lda.score(X_scaled, y_stage))

    scores_df = work_df[["image_id", "fabric", "setting", "group_code", "day", "qc_grade", "exposure_stage"]].copy()
    scores_df["PC1"] = pca_scores[:, 0]
    scores_df["PC2"] = pca_scores[:, 1]
    scores_df["LD1"] = lda_scores[:, 0]
    scores_df["LD2"] = lda_scores[:, 1]

    return {
        "X_scaled": X_scaled,
        "scores_df": scores_df,
        "loadings_df": loadings_df,
        "pca_var_ratio": pca.explained_variance_ratio_,
        "lda_var_ratio": lda.explained_variance_ratio_,
        "lda_accuracy": round(lda_accuracy, 4),
    }


def load_and_aggregate_logger_data(metadata_dir: Path) -> pd.DataFrame:
    """
    Reads INDOOR_DATA_LOGGER.xlsx and OUTDOOR__DATA_LOGGER.xlsx from data/metadata/,
    removes the top summary rows (tem-max / tem-min), parses DD/MM/YYYY + Time timestamps,
    and aggregates hourly readings into 2-day exposure intervals (day = 2, 4, ..., 30)
    to tally with master_features.csv.
    """
    logger_specs = [
        ("Indoor", ["INDOOR_DATA_LOGGER.xlsx"]),
        ("Outdoor", ["OUTDOOR__DATA_LOGGER.xlsx", "OUTDOOR_DATA_LOGGER.xlsx"]),
    ]

    aggregated_frames = []

    for setting_label, candidate_names in logger_specs:
        file_path = next((metadata_dir / name for name in candidate_names if (metadata_dir / name).exists()), None)
        if file_path is None:
            raise FileNotFoundError(f"Missing {setting_label} logger Excel file in {metadata_dir}: tried {candidate_names}")

        raw_df = pd.read_excel(file_path)
        raw_df.columns = [str(c).strip() for c in raw_df.columns]

        # Filter out summary metadata rows ('tem-max', 'tem-min') by keeping numeric 'Number' rows
        valid = raw_df[pd.to_numeric(raw_df["Number"], errors="coerce").notna()].copy()

        # Parse DD/MM/YYYY Date + Time into exact timestamp
        date_str = pd.to_datetime(valid["Date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
        time_str = valid["Time"].astype(str).str.strip()
        valid["timestamp"] = pd.to_datetime(date_str + " " + time_str, errors="coerce")
        valid = valid.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        for col in ["Temperature", "Humidity", "Dew"]:
            valid[col] = pd.to_numeric(valid[col], errors="coerce")

        # Map elapsed hours from deployment (0..720h) into 2-day sampling bins: 2, 4, ..., 30
        t0 = valid["timestamp"].iloc[0]
        elapsed_days = (valid["timestamp"] - t0).dt.total_seconds() / 86400.0
        # Keep readings within the 30-day experimental window (allowing slight clock buffer)
        valid = valid[elapsed_days <= 30.25].copy()
        elapsed_days = elapsed_days.loc[valid.index]

        bin_day = np.ceil(elapsed_days / 2.0).astype(int) * 2
        bin_day = np.clip(bin_day, 2, 30)  # Maps t=0h deployment reading into Day 2 interval
        valid["day"] = bin_day

        # Aggregate per 2-day interval
        grouped = (
            valid.groupby("day", as_index=False)
            .agg(
                n_logger_readings=("Number", "count"),
                interval_temp_mean=("Temperature", "mean"),
                interval_temp_max=("Temperature", "max"),
                interval_rh_mean=("Humidity", "mean"),
                interval_rh_max=("Humidity", "max"),
                interval_dew_mean=("Dew", "mean"),
            )
            .sort_values("day")
        )

        # Also compute cumulative exposure means from Day 0 up to Day d
        grouped["cum_temp_mean"] = grouped["interval_temp_mean"].expanding().mean()
        grouped["cum_rh_mean"] = grouped["interval_rh_mean"].expanding().mean()
        grouped["cum_dew_mean"] = grouped["interval_dew_mean"].expanding().mean()
        grouped.insert(0, "setting", setting_label)

        aggregated_frames.append(grouped.round(3))

    return pd.concat(aggregated_frames, ignore_index=True)


def correlate_environmental_logs(
    master_df: pd.DataFrame,
    logger_summary_df: pd.DataFrame,
    fabric_metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merges aggregated logger data with master_features.csv on ['setting', 'day']
    and computes Pearson (r) and Spearman (rho) correlations against fabric color & texture metrics.
    """
    merged_df = pd.merge(
        master_df[master_df["setting"].isin(["Indoor", "Outdoor"])],
        logger_summary_df,
        on=["setting", "day"],
        how="inner",
    )

    env_cols = [
        "interval_temp_mean", "interval_temp_max",
        "interval_rh_mean", "interval_rh_max", "interval_dew_mean",
        "cum_temp_mean", "cum_rh_mean", "cum_dew_mean",
    ]

    corr_rows = []
    for ev in env_cols:
        for feat in fabric_metrics:
            valid = merged_df[[ev, feat]].dropna()
            if len(valid) >= 5:
                r_val, p_r = pearsonr(valid[ev], valid[feat])
                rho_val, p_rho = spearmanr(valid[ev], valid[feat])
                corr_rows.append({
                    "env_variable": ev,
                    "fabric_feature": feat,
                    "pearson_r": round(float(r_val), 4),
                    "pearson_p": round(float(p_r), 6),
                    "spearman_rho": round(float(rho_val), 4),
                    "spearman_p": round(float(p_rho), 6),
                    "significant_005": bool(p_r < 0.05 or p_rho < 0.05),
                })

    return pd.DataFrame(corr_rows), merged_df
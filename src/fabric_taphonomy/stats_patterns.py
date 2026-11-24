import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def _resolve_logger_path(metadata_dir: Path, base_name: str) -> Path | None:
    """Finds INDOOR / OUTDOOR logger files across .xls, .xlsx, .xslx, and .csv extensions."""
    for ext in [".xls", ".xlsx", ".xslx", ".csv"]:
        for name_variant in (base_name, base_name.lower(), base_name.capitalize()):
            candidate = metadata_dir / f"{name_variant}{ext}"
            if candidate.exists():
                return candidate
    return None


def _load_logger_dataframe(file_path: Path) -> pd.DataFrame:
    """Loads .xls, .xlsx, .xslx, or disguised TSV/CSV logger files automatically."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xslx"}:
        return pd.read_excel(file_path, engine="openpyxl")
    if suffix == ".xls":
        try:
            return pd.read_excel(file_path, engine="xlrd")
        except Exception:
            # Fallback if USB logger software saved tab/comma-separated text with a .xls extension
            for sep in ["\t", ",", ";"]:
                try:
                    df_txt = pd.read_csv(file_path, sep=sep)
                    if df_txt.shape[1] >= 2:
                        return df_txt
                except Exception:
                    continue
    return pd.read_excel(file_path)


def _parse_single_logger(file_path: Path, setting_label: str) -> pd.DataFrame:
    """
    Reads an hourly data logger file, identifies Temperature (°C) and Relative Humidity (%)
    columns automatically, groups hourly readings into Exposure Days (Days 1..30), and
    computes both 24-hr daily means and 48-hr rolling exposure window means.
    """
    raw_log = _load_logger_dataframe(file_path)
    raw_log.columns = [str(c).strip() for c in raw_log.columns]

    # Identify Temperature and Relative Humidity columns flexibly
    temp_col = next((c for c in raw_log.columns if "temp" in c.lower() or "°c" in c.lower() or c.lower() == "t"), None)
    rh_col = next((c for c in raw_log.columns if "humid" in c.lower() or "rh" in c.lower() or "%" in c.lower()), None)
    day_col = next((c for c in raw_log.columns if c.lower() in {"day", "exposure_day", "days"}), None)

    if temp_col is None or rh_col is None:
        raise ValueError(
            f"Could not detect Temperature and Humidity columns in {file_path.name}. "
            f"Found columns: {list(raw_log.columns)}"
        )

    df = pd.DataFrame({
        "temp_c": pd.to_numeric(raw_log[temp_col], errors="coerce"),
        "rh_pct": pd.to_numeric(raw_log[rh_col], errors="coerce"),
    }).dropna().reset_index(drop=True)

    # Determine Exposure Day (1 to 30) safely without DD/MM vs MM/DD swap issues
    if day_col is not None:
        df["day"] = pd.to_numeric(raw_log.loc[df.index, day_col], errors="coerce").astype(int)
    else:
        date_col = next((c for c in raw_log.columns if c.lower() == "date"), None)
        time_only_col = next((c for c in raw_log.columns if c.lower() == "time"), None)
        combined_time_col = next((c for c in raw_log.columns if any(k in c.lower() for k in ["time", "date", "stamp"])), None)

        if date_col is not None and time_only_col is not None:
            raw_dt_series = raw_log.loc[df.index, date_col].astype(str) + " " + raw_log.loc[df.index, time_only_col].astype(str)
        elif combined_time_col is not None:
            raw_dt_series = raw_log.loc[df.index, combined_time_col]
        else:
            raw_dt_series = None

        if raw_dt_series is not None:
            sample_str = str(raw_dt_series.dropna().iloc[0]).strip()
            use_dayfirst = "/" in sample_str and len(sample_str.split("/")[0]) <= 2

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                dt = pd.to_datetime(raw_dt_series, dayfirst=use_dayfirst, errors="coerce")

            if dt.notna().sum() > 0:
                start_date = dt.dropna().iloc[0].normalize()
                computed_days = ((dt.dt.normalize() - start_date).dt.days + 1).clip(lower=1)
                # Guard against DD/MM swap producing >60 days on a 30-day study
                if computed_days.max() <= 60:
                    df["day"] = computed_days.astype(int)
                else:
                    df["day"] = (df.index // 24) + 1
            else:
                df["day"] = (df.index // 24) + 1
        else:
            df["day"] = (df.index // 24) + 1

    daily = (
        df.groupby("day", as_index=False)
        .agg(
            daily_temp_mean=("temp_c", "mean"),
            daily_temp_max=("temp_c", "max"),
            daily_temp_min=("temp_c", "min"),
            daily_rh_mean=("rh_pct", "mean"),
            daily_rh_max=("rh_pct", "max"),
            daily_rh_min=("rh_pct", "min"),
        )
        .sort_values("day")
    )

    # Compute 48-hour interval mean (matching the 2-day sampling interval)
    daily["temp_48hr_mean"] = daily["daily_temp_mean"].rolling(window=2, min_periods=1).mean().round(3)
    daily["rh_48hr_mean"] = daily["daily_rh_mean"].rolling(window=2, min_periods=1).mean().round(3)
    daily["daily_temp_mean"] = daily["daily_temp_mean"].round(3)
    daily["daily_rh_mean"] = daily["daily_rh_mean"].round(3)
    daily["setting"] = setting_label

    # Add Day 0 baseline row using Day 1 initial condition
    day0_row = daily.iloc[[0]].copy()
    day0_row["day"] = 0
    daily_full = pd.concat([day0_row, daily], ignore_index=True)

    return daily_full


def merge_environmental_logs(master_df: pd.DataFrame, metadata_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads INDOOR.xslx and OUTDOOR.xslx from data/metadata/ and merges 48-hr and daily
    microclimate metrics onto master_features.csv by setting and exposure day.
    """
    indoor_path = _resolve_logger_path(metadata_dir, "INDOOR")
    outdoor_path = _resolve_logger_path(metadata_dir, "OUTDOOR")

    if indoor_path is None or outdoor_path is None:
        raise FileNotFoundError(
            f"Missing logger files in {metadata_dir}. "
            f"Found INDOOR={indoor_path}, OUTDOOR={outdoor_path}."
        )

    indoor_daily = _parse_single_logger(indoor_path, "Indoor")
    outdoor_daily = _parse_single_logger(outdoor_path, "Outdoor")
    env_summary_df = pd.concat([indoor_daily, outdoor_daily], ignore_index=True)

    merged_df = master_df.copy()
    # Map Control samples (Day 0) to Indoor baseline microclimate for merging
    merged_df["_env_setting"] = merged_df["setting"].replace({"Control": "Indoor"})

    merged_df = merged_df.merge(
        env_summary_df[["setting", "day", "daily_temp_mean", "daily_rh_mean", "temp_48hr_mean", "rh_48hr_mean"]],
        left_on=["_env_setting", "day"],
        right_on=["setting", "day"],
        how="left",
        suffixes=("", "_env"),
    )
    merged_df.drop(columns=["_env_setting", "setting_env"], inplace=True, errors="ignore")
    return merged_df, env_summary_df


def compute_env_correlations(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Computes Pearson (r) and Spearman (rho) correlations between Temp/RH and image features."""
    sub = df.dropna(subset=["temp_48hr_mean", "rh_48hr_mean"]).copy()
    rows = []

    for feat in feature_cols:
        r_temp, p_r_temp = stats.pearsonr(sub["temp_48hr_mean"], sub[feat])
        rho_temp, p_rho_temp = stats.spearmanr(sub["temp_48hr_mean"], sub[feat])
        r_rh, p_r_rh = stats.pearsonr(sub["rh_48hr_mean"], sub[feat])
        rho_rh, p_rho_rh = stats.spearmanr(sub["rh_48hr_mean"], sub[feat])

        rows.append({
            "feature": feat,
            "pearson_r_temp": round(float(r_temp), 4),
            "p_pearson_temp": round(float(p_r_temp), 5),
            "spearman_rho_temp": round(float(rho_temp), 4),
            "p_spearman_temp": round(float(p_rho_temp), 5),
            "pearson_r_rh": round(float(r_rh), 4),
            "p_pearson_rh": round(float(p_r_rh), 5),
            "spearman_rho_rh": round(float(rho_rh), 4),
            "p_spearman_rh": round(float(p_rho_rh), 5),
        })

    return pd.DataFrame(rows)


def run_two_way_anova_suite(df: pd.DataFrame, metrics: list[str], alpha: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs Two-Way ANOVA (Type II SS) for:
      Model A: Fabric Type x Exposure Setting (Indoor vs. Outdoor)
      Model B: Fabric Type x Exposure Duration (Days)
    Plus Tukey's HSD post-hoc comparisons across Fabric x Setting groups.
    """
    sub = df[df["setting"].isin(["Indoor", "Outdoor"])].copy()
    sub["fabric_setting"] = sub["fabric"] + " (" + sub["setting"] + ")"

    anova_rows = []
    tukey_rows = []

    for metric in metrics:
        # Model A: Fabric x Setting
        model_fs = ols(f"{metric} ~ C(fabric) + C(setting) + C(fabric):C(setting)", data=sub).fit()
        aov_fs = sm.stats.anova_lm(model_fs, typ=2)

        for term, label in [
            ("C(fabric)", "Fabric Type"),
            ("C(setting)", "Exposure Setting"),
            ("C(fabric):C(setting)", "Fabric x Setting Interaction"),
        ]:
            f_val = float(aov_fs.loc[term, "F"])
            p_val = float(aov_fs.loc[term, "PR(>F)"])
            anova_rows.append({
                "metric": metric,
                "design": "Fabric x Setting",
                "factor": label,
                "F_stat": round(f_val, 4),
                "p_value": round(p_val, 6),
                "significant_0.05": p_val < alpha,
            })

        # Model B: Fabric x Exposure Day
        model_fd = ols(f"{metric} ~ C(fabric) + C(day) + C(fabric):C(day)", data=sub).fit()
        aov_fd = sm.stats.anova_lm(model_fd, typ=2)

        for term, label in [
            ("C(fabric)", "Fabric Type"),
            ("C(day)", "Exposure Duration (Day)"),
            ("C(fabric):C(day)", "Fabric x Day Interaction"),
        ]:
            f_val = float(aov_fd.loc[term, "F"])
            p_val = float(aov_fd.loc[term, "PR(>F)"])
            anova_rows.append({
                "metric": metric,
                "design": "Fabric x Exposure Day",
                "factor": label,
                "F_stat": round(f_val, 4),
                "p_value": round(p_val, 6),
                "significant_0.05": p_val < alpha,
            })

        # Tukey's HSD across the 4 Fabric x Setting combinations
        tukey = pairwise_tukeyhsd(endog=sub[metric], groups=sub["fabric_setting"], alpha=alpha)
        tukey_tbl = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        tukey_tbl.insert(0, "metric", metric)
        tukey_rows.append(tukey_tbl)

    return pd.DataFrame(anova_rows), pd.concat(tukey_rows, ignore_index=True)


def fit_degradation_kinetics(df: pd.DataFrame, metric: str) -> tuple[dict, pd.DataFrame]:
    """
    Fits asymptotic exponential saturation curves y(t) = A * (1 - exp(-k * t)) + y0
    (with linear/quadratic fallback for flat indoor baselines) across Days 0..30
    for each Fabric x Setting trajectory.
    """
    def exp_saturation(t, amp, k, y0):
        return amp * (1.0 - np.exp(-k * t)) + y0

    curves = {}
    summary_rows = []

    for fab in ["Cotton", "Polyester"]:
        ctrl_vals = df[(df["fabric"] == fab) & (df["setting"] == "Control")][["day", metric]]
        for setting in ["Indoor", "Outdoor"]:
            grp = df[(df["fabric"] == fab) & (df["setting"] == setting)][["day", metric]]
            traj = pd.concat([ctrl_vals, grp], ignore_index=True).sort_values("day")

            t_data = traj["day"].to_numpy(dtype=float)
            y_data = traj[metric].to_numpy(dtype=float)
            t_grid = np.linspace(0, t_data.max(), 100)

            try:
                y0_guess = float(np.mean(y_data[t_data == 0])) if np.any(t_data == 0) else float(y_data[0])
                amp_guess = max(1.0, float(np.max(y_data) - y0_guess))
                popt, _ = curve_fit(
                    exp_saturation,
                    t_data,
                    y_data,
                    p0=[amp_guess, 0.10, y0_guess],
                    bounds=([-100.0, 0.001, -50.0], [300.0, 2.0, 300.0]),
                    maxfev=5000,
                )
                y_pred = exp_saturation(t_data, *popt)
                y_smooth = exp_saturation(t_grid, *popt)
                ss_res = np.sum((y_data - y_pred) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))
                model_type = "Exponential Saturation"
                rate_k = round(float(popt[1]), 4)
            except Exception:
                # Linear regression fallback when indoor curve shows negligible drift
                slope, intercept, r_val, _, _ = stats.linregress(t_data, y_data)
                y_smooth = slope * t_grid + intercept
                r2 = float(r_val ** 2)
                model_type = "Linear Trend"
                rate_k = round(float(slope), 4)

            label = f"{fab} ({setting})"
            curves[label] = {"t_data": t_data, "y_data": y_data, "t_grid": t_grid, "y_smooth": y_smooth}
            summary_rows.append({
                "metric": metric,
                "trajectory": label,
                "model_type": model_type,
                "rate_constant_k_or_slope": rate_k,
                "R_squared": round(r2, 4),
            })

    return curves, pd.DataFrame(summary_rows)


def run_multivariate_pca_lda(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Standardizes combined color and texture features (Z-score scaling) and performs:
      1. Principal Component Analysis (PCA) for unsupervised variance & feature loadings.
      2. Linear Discriminant Analysis (LDA) for supervised separation across exposure phases:
         - Stage 0: Unexposed Control (Day 0)
         - Stage 1: Early Exposure (Days 1–8)
         - Stage 2: Mid Exposure (Days 10–20)
         - Stage 3: Late Exposure (Days 22–30)
    """
    X = df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. PCA
    pca = PCA()
    pca_scores = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_ * 100.0
    loadings_df = pd.DataFrame(
        pca.components_[:3].T,
        index=feature_cols,
        columns=["PC1_loading", "PC2_loading", "PC3_loading"],
    ).round(4)

    # 2. Define Exposure Phase Labels for Supervised LDA
    def assign_stage(row):
        if row["day"] == 0:
            return "Control (Day 0)"
        elif row["day"] <= 8:
            return "Early (Days 1-8)"
        elif row["day"] <= 20:
            return "Mid (Days 10-20)"
        else:
            return "Late (Days 22-30)"

    stages = df.apply(assign_stage, axis=1)
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_scores = lda.fit_transform(X_scaled, stages)
    lda_var = lda.explained_variance_ratio_ * 100.0

    return {
        "X_scaled": X_scaled,
        "pca_scores": pca_scores,
        "pca_explained_var": explained_var,
        "loadings_df": loadings_df,
        "stages": stages,
        "lda_scores": lda_scores,
        "lda_explained_var": lda_var,
    }
# =============================================================================
# CT032-3-3 Further Artificial Intelligence (FAI) — Part 2
# Walmart Weekly Sales Forecasting — Streamlit GUI
# Group H | APD3F2511CS(AI)
# =============================================================================

import os
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")
sns.set_palette("muted")
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales Forecasting | Group H",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "FAI Dataset")
TARGET     = "Weekly_Sales"
FEATURES_A = [
    "Store", "Dept", "year", "month", "week",
    "IsHoliday", "Type", "Size",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
]
MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
FEATURES_B    = FEATURES_A + MARKDOWN_COLS + [f"{c}_missing" for c in MARKDOWN_COLS]

# Store metadata (from stores.csv — used to auto-fill Type & Size in forecast form)
STORE_INFO = {
    1: ("A", 151315),  2: ("A", 202307),  3: ("B",  37392),  4: ("A", 205863),
    5: ("B",  34875),  6: ("A", 202505),  7: ("B",  70713),  8: ("A", 155078),
    9: ("B", 125833), 10: ("B", 126512), 11: ("A", 207499), 12: ("B", 112238),
   13: ("A", 219622), 14: ("A", 200898), 15: ("B", 123737), 16: ("B",  57197),
   17: ("B",  93188), 18: ("B", 120653), 19: ("A", 203819), 20: ("A", 203742),
   21: ("B", 140167), 22: ("B", 119557), 23: ("B", 114533), 24: ("A", 203819),
   25: ("B", 128107), 26: ("A", 152513), 27: ("A", 204184), 28: ("A", 206302),
   29: ("B",  93638), 30: ("C",  42988), 31: ("A", 203750), 32: ("A", 203007),
   33: ("A",  39690), 34: ("A", 158114), 35: ("B", 103681), 36: ("A",  39910),
   37: ("C",  39910), 38: ("C",  39690), 39: ("A", 184109), 40: ("A", 155083),
   41: ("A", 196321), 42: ("C",  39690), 43: ("C",  41062), 44: ("C",  39910),
   45: ("B", 118221),
}
TYPE_ENCODE = {"A": 0, "B": 1, "C": 2}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Return MAE, RMSE, WMAPE (scale-weighted MAPE, stable on near-zero rows)."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true > 0
    wmape = np.sum(np.abs(y_true[mask] - y_pred[mask])) / np.sum(y_true[mask]) * 100
    return mae, rmse, wmape


# ─── Pipeline: data loading + model training (cached once per session) ────────

def _models_saved():
    """Return True if all files exported from Colab are present."""
    files = [
        "xgb_model_A.json", "xgb_model_B.json", "rf_importances.npy",
        "y_test.npy", "y_pred_xgb_A.npy", "y_pred_rf_A.npy",
        "y_pred_xgb_B.npy", "test_holiday.npy", "test_dates.csv",
    ]
    return all(os.path.exists(os.path.join(BASE_DIR, f)) for f in files)


@st.cache_resource(show_spinner="Loading data and models — please wait...")
def build_pipeline():
    # --- Load raw CSVs (always needed for EDA, forecast history, data overview) ---
    train    = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    features = pd.read_csv(os.path.join(DATA_PATH, "features.csv"))
    stores   = pd.read_csv(os.path.join(DATA_PATH, "stores.csv"))

    # --- Merge ---
    df = train.merge(features, on=["Store", "Date"], how="left", suffixes=("", "_feat"))
    df = df.merge(stores, on="Store", how="left")
    if "IsHoliday_feat" in df.columns:
        df.drop(columns=["IsHoliday_feat"], inplace=True)

    # --- Preprocessing ---
    df["Date"]      = pd.to_datetime(df["Date"])
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    le              = LabelEncoder()
    df["Type"]      = le.fit_transform(df["Type"])   # A=0, B=1, C=2
    for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)

    # --- Feature engineering ---
    df["year"]  = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["week"]  = df["Date"].dt.isocalendar().week.astype(int)
    df.sort_values(["Store", "Dept", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Scenario B ---
    df_b = df.copy()
    for col in MARKDOWN_COLS:
        if col in df_b.columns:
            df_b[f"{col}_missing"] = df_b[col].isnull().astype(int)
            df_b[col].fillna(0, inplace=True)

    # --- Chronological 80/20 split ---
    unique_dates = sorted(df["Date"].unique())
    cutoff_date  = unique_dates[int(len(unique_dates) * 0.80)]
    train_mask   = df["Date"] < cutoff_date
    test_mask    = df["Date"] >= cutoff_date

    if _models_saved():
        # ── Load pre-trained models + predictions exported from Colab ──────────
        # This guarantees numbers match the documentation exactly.
        # Use xgb.Booster (version-agnostic) to avoid sklearn wrapper type errors.
        xgb_model = xgb.Booster()
        xgb_model.load_model(os.path.join(BASE_DIR, "xgb_model_A.json"))

        xgb_model_B = xgb.Booster()
        xgb_model_B.load_model(os.path.join(BASE_DIR, "xgb_model_B.json"))

        rf_importances = np.load(os.path.join(BASE_DIR, "rf_importances.npy"))
        rf_model = None   # full model not needed; importances loaded separately

        y_test       = np.load(os.path.join(BASE_DIR, "y_test.npy"))
        y_pred_xgb_A = np.load(os.path.join(BASE_DIR, "y_pred_xgb_A.npy"))
        y_pred_rf_A  = np.load(os.path.join(BASE_DIR, "y_pred_rf_A.npy"))
        y_pred_xgb_B = np.load(os.path.join(BASE_DIR, "y_pred_xgb_B.npy"))
        test_holiday = np.load(os.path.join(BASE_DIR, "test_holiday.npy"))
        test_dates   = pd.read_csv(os.path.join(BASE_DIR, "test_dates.csv"),
                                   parse_dates=["Date"])["Date"]
        X_test_A = df.loc[test_mask, FEATURES_A]

    else:
        # ── Fallback: train locally (used before Colab files are added) ─────────
        X_train_A = df.loc[train_mask, FEATURES_A]
        y_train   = df.loc[train_mask, TARGET]
        X_test_A  = df.loc[test_mask,  FEATURES_A]
        y_test    = df.loc[test_mask,  TARGET].values
        X_train_B = df_b.loc[train_mask, FEATURES_B]
        X_test_B  = df_b.loc[test_mask,  FEATURES_B]
        test_dates   = df.loc[test_mask, "Date"]
        test_holiday = df.loc[test_mask, "IsHoliday"].values

        _xgb_A = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5,
            reg_alpha=0.1, min_child_weight=5,
            objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0,
        )
        _xgb_A.fit(X_train_A, y_train)
        y_pred_xgb_A = np.clip(_xgb_A.predict(X_test_A), 0, None)
        xgb_model = _xgb_A.get_booster()   # normalise to Booster like the load path

        rf_model = RandomForestRegressor(
            n_estimators=200, max_depth=12, max_features=0.6,
            min_samples_leaf=4, random_state=42, n_jobs=-1,
        )
        rf_model.fit(X_train_A, y_train)
        y_pred_rf_A    = np.clip(rf_model.predict(X_test_A), 0, None)
        rf_importances = rf_model.feature_importances_

        _xgb_B = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5,
            reg_alpha=0.1, min_child_weight=5,
            objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0,
        )
        _xgb_B.fit(X_train_B, y_train)
        y_pred_xgb_B = np.clip(_xgb_B.predict(X_test_B), 0, None)
        xgb_model_B = _xgb_B.get_booster()

    return {
        "df": df, "df_b": df_b, "le": le,
        "train_mask": train_mask, "test_mask": test_mask,
        "cutoff_date": cutoff_date,
        "X_test_A": X_test_A,
        "y_test": y_test,
        "y_pred_xgb_A": y_pred_xgb_A,
        "y_pred_rf_A":  y_pred_rf_A,
        "y_pred_xgb_B": y_pred_xgb_B,
        "test_dates":   test_dates,
        "test_holiday": test_holiday,
        "xgb_model":       xgb_model,
        "rf_model":        rf_model,
        "rf_importances":  rf_importances,
        "xgb_model_B":     xgb_model_B,
    }


# =============================================================================
# SIDEBAR — Navigation
# =============================================================================

st.sidebar.title("Walmart Sales Forecasting")
st.sidebar.caption("CT032-3-3 FAI — Part 2 | Group H")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Data Overview",
        "Exploratory Analysis",
        "Model Evaluation",
        "Visualizations",
        "Interactive Forecast",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Group H — APD3F2511CS(AI)**\n\n"
    "- Sanjivan (TP070073)\n"
    "- Devara (TP073570)\n"
    "- Jiyad (TP077380)\n"
    "- Taneshen (TP078396)\n"
    "- Fareez (TP077930)\n\n"
    "*Lecturer: Dr. Adeline Sneha John Chrisastum*"
)

# =============================================================================
# Load pipeline (triggers on first page visit)
# =============================================================================

p = build_pipeline()

df          = p["df"]
train_mask  = p["train_mask"]
test_mask   = p["test_mask"]
cutoff_date = p["cutoff_date"]
y_test      = p["y_test"]
y_pred_xgb_A = p["y_pred_xgb_A"]
y_pred_rf_A  = p["y_pred_rf_A"]
y_pred_xgb_B = p["y_pred_xgb_B"]
test_dates   = p["test_dates"]
test_holiday = p["test_holiday"]
xgb_model    = p["xgb_model"]
rf_model        = p["rf_model"]
rf_importances  = p["rf_importances"]
X_test_A     = p["X_test_A"]

mae_xgb_A,  rmse_xgb_A,  wmape_xgb_A  = compute_metrics(y_test, y_pred_xgb_A)
mae_rf_A,   rmse_rf_A,   wmape_rf_A   = compute_metrics(y_test, y_pred_rf_A)
mae_xgb_B,  rmse_xgb_B,  wmape_xgb_B  = compute_metrics(y_test, y_pred_xgb_B)


# =============================================================================
# PAGE 1 — Overview
# =============================================================================

if page == "Overview":
    st.title("Walmart Weekly Sales Forecasting")
    st.subheader("CT032-3-3 Further Artificial Intelligence — Part 2 Implementation")

    st.markdown("""
    This application implements the machine learning solution designed in Part 1 for
    predicting weekly retail sales across Walmart stores and departments.

    ---
    ### Pipeline Architecture

    | Step | Description |
    |------|-------------|
    | 1. Data Ingestion | Load and merge `train.csv`, `features.csv`, `stores.csv` |
    | 2. Preprocessing | Handle missing values, encode categoricals |
    | 3. Feature Engineering | Extract temporal features (year, month, week) from Date |
    | 4. Feature Selection | Scenario A (leakage-safe) as primary |
    | 5. Model Training | XGBoost Regressor (primary) + Random Forest (validation) |
    | 6. Evaluation | MAE, RMSE, WMAPE with holiday vs non-holiday breakdown |
    | 7. Visualization | Actual vs Predicted, feature importance, SHAP analysis |
    | 8. Scenario B | Secondary comparison including MarkDown variables |

    ---
    ### Key Design Decisions

    - **Chronological split**: 80% train / 20% test by unique week dates — no data leakage
    - **Scenario A (primary)**: Excludes MarkDown features (>60% missing) — leakage-safe
    - **Scenario B (secondary)**: Zero imputation + missingness flags for MarkDowns
    - **Negative clip**: All predictions clipped to 0 (sales cannot be negative)
    - **WMAPE over MAPE**: ~6.4% of rows have Weekly_Sales < $100; WMAPE is scale-weighted
      and avoids inflation from near-zero denominators
    """)

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Unique Stores", df["Store"].nunique())
    col3.metric("Unique Departments", df["Dept"].nunique())
    col4.metric(
        "Date Range",
        f"{df['Date'].min().date()} → {df['Date'].max().date()}",
    )

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Primary Model: XGBoost Scenario A**")
        st.metric("MAE",   f"${mae_xgb_A:,.0f}")
        st.metric("RMSE",  f"${rmse_xgb_A:,.0f}")
        st.metric("WMAPE", f"{wmape_xgb_A:.2f}%")
    with col_b:
        st.markdown("**Scenario A Features (12)**")
        st.code(", ".join(FEATURES_A))


# =============================================================================
# PAGE 2 — Data Overview
# =============================================================================

elif page == "Data Overview":
    st.title("Data Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Train rows", f"{train_mask.sum():,}")
    col2.metric("Test rows",  f"{test_mask.sum():,}")
    col3.metric(
        "Train/Test cutoff",
        str(pd.Timestamp(cutoff_date).date()),
    )

    st.markdown("---")
    st.subheader("Sample Records")
    st.dataframe(
        df[["Store", "Dept", "Date", "IsHoliday", "Type", "Size",
            "Temperature", "Fuel_Price", "CPI", "Unemployment", "Weekly_Sales"]]
        .head(20),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Missing Values per Feature")
    all_feat_cols = [
        "Store", "Dept", "Type", "Size", "IsHoliday",
        "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    ]
    miss_data = {
        col: round(df[col].isnull().sum() / len(df) * 100, 2)
        for col in all_feat_cols if col in df.columns
    }
    miss_df = pd.DataFrame(
        miss_data.items(), columns=["Feature", "Missing (%)"]
    ).sort_values("Missing (%)", ascending=False)
    st.dataframe(miss_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Descriptive Statistics — Weekly Sales")
    st.dataframe(
        df["Weekly_Sales"].describe().rename("Weekly_Sales").to_frame().T,
        use_container_width=True,
    )

    st.subheader("Store Type Distribution")
    type_labels = {0: "A", 1: "B", 2: "C"}
    type_counts = df["Store"].drop_duplicates().map(
        lambda s: STORE_INFO[s][0]
    ).value_counts().sort_index()
    st.bar_chart(type_counts)


# =============================================================================
# PAGE 3 — Exploratory Analysis
# =============================================================================

elif page == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")

    type_map_inv = {0: "A", 1: "B", 2: "C"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Exploratory Data Analysis — Walmart Weekly Sales", fontsize=14, fontweight="bold")

    # Plot 1: Mean Weekly Sales Over Time
    ax1 = axes[0, 0]
    weekly_mean = df.groupby("Date")["Weekly_Sales"].mean()
    ax1.plot(weekly_mean.index, weekly_mean.values, color="steelblue", linewidth=1.2)
    ax1.set_title("Mean Weekly Sales Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Mean Weekly Sales ($)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Plot 2: Holiday vs Non-Holiday
    ax2 = axes[0, 1]
    holiday_mean = df.groupby("IsHoliday")["Weekly_Sales"].mean()
    bars = ax2.bar(
        ["Non-Holiday", "Holiday"], holiday_mean.values,
        color=["#4c9be8", "#e87b4c"], edgecolor="white",
    )
    ax2.set_title("Holiday vs Non-Holiday Effect (Mean Weekly Sales)")
    ax2.set_ylabel("Mean Weekly Sales ($)")
    for bar, val in zip(bars, holiday_mean.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"${val:,.0f}", ha="center", va="bottom", fontweight="bold",
        )

    # Plot 3: Missingness per Feature
    ax3 = axes[1, 0]
    all_feat_cols = [
        "Store", "Dept", "Type", "Size", "IsHoliday",
        "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    ]
    miss_pct = [
        (df[c].isnull().sum() / len(df) * 100) if c in df.columns else 0
        for c in all_feat_cols
    ]
    colors_miss = ["#e05c5c" if m > 10 else "#4cae4c" for m in miss_pct]
    ax3.barh(all_feat_cols, miss_pct, color=colors_miss)
    ax3.set_title("Missingness (%) per Feature")
    ax3.set_xlabel("% Missing")
    ax3.axvline(10, color="red", linestyle="--", alpha=0.6, label="10% threshold")
    ax3.legend()

    # Plot 4: Sales Distribution by Store Type
    ax4 = axes[1, 1]
    for t_val, t_name in type_map_inv.items():
        subset = df[df["Type"] == t_val]["Weekly_Sales"]
        ax4.hist(subset, bins=60, alpha=0.6, label=f"Type {t_name}", density=True)
    ax4.set_title("Weekly Sales Distribution by Store Type")
    ax4.set_xlabel("Weekly Sales ($)")
    ax4.set_ylabel("Density")
    ax4.set_xlim(-5000, 150000)
    ax4.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Top-left: seasonal demand peaks visible in Nov–Dec each year. "
        "Top-right: holiday weeks average ~8% higher sales. "
        "Bottom-left: MarkDown features are heavily missing (>60%) — excluded from Scenario A. "
        "Bottom-right: Type A stores (largest) show a wider, flatter sales distribution."
    )


# =============================================================================
# PAGE 4 — Model Evaluation
# =============================================================================

elif page == "Model Evaluation":
    st.title("Model Evaluation")

    # Summary metrics table
    st.subheader("Performance on Test Set")
    results_df = pd.DataFrame({
        "Model": [
            "XGBoost — Scenario A (Primary)",
            "Random Forest — Scenario A (Validation)",
            "XGBoost — Scenario B (MarkDowns)",
        ],
        "MAE ($)":   [f"{mae_xgb_A:,.2f}",  f"{mae_rf_A:,.2f}",  f"{mae_xgb_B:,.2f}"],
        "RMSE ($)":  [f"{rmse_xgb_A:,.2f}", f"{rmse_rf_A:,.2f}", f"{rmse_xgb_B:,.2f}"],
        "WMAPE (%)": [f"{wmape_xgb_A:.2f}", f"{wmape_rf_A:.2f}", f"{wmape_xgb_B:.2f}"],
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Holiday breakdown
    st.subheader("Holiday vs Non-Holiday Breakdown — XGBoost Scenario A")
    hol_idx     = test_holiday == 1
    non_hol_idx = test_holiday == 0

    if hol_idx.sum() > 0:
        mae_h, rmse_h, wmape_h       = compute_metrics(y_test[hol_idx],     y_pred_xgb_A[hol_idx])
        mae_nh, rmse_nh, wmape_nh    = compute_metrics(y_test[non_hol_idx], y_pred_xgb_A[non_hol_idx])
        holiday_df = pd.DataFrame({
            "Period":    ["Holiday weeks", "Non-Holiday weeks"],
            "Rows":      [int(hol_idx.sum()), int(non_hol_idx.sum())],
            "MAE ($)":   [f"{mae_h:,.2f}",  f"{mae_nh:,.2f}"],
            "RMSE ($)":  [f"{rmse_h:,.2f}", f"{rmse_nh:,.2f}"],
            "WMAPE (%)": [f"{wmape_h:.2f}", f"{wmape_nh:.2f}"],
        })
        st.dataframe(holiday_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # MAE/RMSE bar chart
    st.subheader("MAE & RMSE Comparison")
    models    = ["XGBoost\nScenario A", "Random Forest\nScenario A", "XGBoost\nScenario B"]
    mae_vals  = [mae_xgb_A,  mae_rf_A,  mae_xgb_B]
    rmse_vals = [rmse_xgb_A, rmse_rf_A, rmse_xgb_B]
    x     = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width / 2, mae_vals,  width, label="MAE",  color="steelblue",  alpha=0.85)
    b2 = ax.bar(x + width / 2, rmse_vals, width, label="RMSE", color="darkorange", alpha=0.85)
    for bar in list(b1) + list(b2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f"${bar.get_height():,.0f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Error ($)")
    ax.set_title("Model Performance Comparison — MAE & RMSE")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "XGBoost Scenario A achieves the lowest MAE and RMSE. "
        "Scenario B (MarkDowns included) shows marginal improvement, "
        "but zero-imputation of heavily missing MarkDown values limits its benefit."
    )


# =============================================================================
# PAGE 5 — Visualizations
# =============================================================================

elif page == "Visualizations":
    st.title("Visualizations")

    test_df_plot = pd.DataFrame({
        "Date":      test_dates.values,
        "Actual":    y_test,
        "Predicted": y_pred_xgb_A,
        "IsHoliday": test_holiday,
    })
    weekly_actual = test_df_plot.groupby("Date")["Actual"].mean()
    weekly_pred   = test_df_plot.groupby("Date")["Predicted"].mean()
    weekly_hol    = test_df_plot.groupby("Date")["IsHoliday"].max()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Actual vs Predicted",
        "Scatter Plot",
        "Feature Importance",
        "Residuals",
        "SHAP",
    ])

    # ── Tab 1: Actual vs Predicted ──
    with tab1:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(weekly_actual.index, weekly_actual.values,
                label="Actual", color="steelblue", linewidth=1.5)
        ax.plot(weekly_pred.index, weekly_pred.values,
                label="XGBoost Predicted", color="orange", linewidth=1.5, linestyle="--")
        for date, is_hol in weekly_hol.items():
            if is_hol:
                ax.axvline(date, color="red", alpha=0.15, linewidth=6)
        ax.set_title(
            "Actual vs Predicted Weekly Sales (Test Set)\nRed bands = Holiday weeks",
            fontsize=12,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Mean Weekly Sales ($)")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Mean actual vs predicted weekly sales over the test period. "
            "Red bands highlight holiday weeks where demand spikes are observed. "
            "The model closely tracks the actual trend, with slight under-prediction during peak holidays."
        )

    # ── Tab 2: Scatter ──
    with tab2:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(y_test), size=min(5000, len(y_test)), replace=False)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(
            y_test[sample_idx], y_pred_xgb_A[sample_idx],
            alpha=0.25, s=8, color="steelblue",
        )
        max_val = max(y_test[sample_idx].max(), y_pred_xgb_A[sample_idx].max())
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_title(f"Actual vs Predicted Scatter (n=5,000 sample)\nMAE = ${mae_xgb_A:,.0f}")
        ax.set_xlabel("Actual Weekly Sales ($)")
        ax.set_ylabel("Predicted Weekly Sales ($)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Each point represents one Store/Dept/week prediction. "
            "Points on the red diagonal indicate perfect predictions. "
            "Dense clustering along the diagonal shows strong model accuracy."
        )

    # ── Tab 3: Feature Importance ──
    with tab3:
        xgb_imp = (
            pd.Series(
                xgb_model.get_score(importance_type="gain"),
                name="XGBoost Gain",
            )
            .reindex(FEATURES_A)
            .fillna(0)
            .sort_values(ascending=True)
        )
        rf_imp = pd.Series(
            rf_importances, index=FEATURES_A
        ).sort_values(ascending=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Feature Importance Comparison", fontsize=13, fontweight="bold")
        axes[0].barh(xgb_imp.index, xgb_imp.values, color="steelblue")
        axes[0].set_title("XGBoost — Feature Importance (Gain)")
        axes[0].set_xlabel("Gain Score")
        axes[1].barh(rf_imp.index, rf_imp.values, color="darkorange")
        axes[1].set_title("Random Forest — Feature Importance (Gini)")
        axes[1].set_xlabel("Gini Score")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Both XGBoost (gain-based) and Random Forest (Gini) consistently rank "
            "Dept and Size as the most important features, validating the Part 1 feature selection. "
            "Temporal features (week, month, year) also rank highly, confirming seasonality."
        )

    # ── Tab 4: Residuals ──
    with tab4:
        residuals = y_test - y_pred_xgb_A

        wr = test_df_plot.copy()
        wr["Residual"] = residuals
        wr = wr.groupby("Date")["Residual"].mean()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(residuals, bins=80, color="steelblue", alpha=0.8, edgecolor="white")
        axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
        axes[0].axvline(
            residuals.mean(), color="orange", linestyle="--", linewidth=1.5,
            label=f"Mean = ${residuals.mean():,.0f}",
        )
        axes[0].set_title("Residual Distribution")
        axes[0].set_xlabel("Residual ($)")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        axes[1].plot(wr.index, wr.values, color="steelblue", linewidth=1)
        axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
        axes[1].set_title("Mean Residuals Over Time")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Mean Residual ($)")
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Residuals centred near zero indicate an unbiased model. "
            "Slight positive skew reflects under-prediction during holiday sales spikes."
        )

    # ── Tab 5: SHAP ──
    with tab5:
        if not SHAP_AVAILABLE:
            st.warning("SHAP library is not installed. Run `pip install shap` to enable this view.")
        else:
            with st.spinner("Computing SHAP values (sampling 2,000 rows)..."):
                rng2       = np.random.default_rng(0)
                shap_idx   = rng2.choice(len(X_test_A), size=min(2000, len(X_test_A)), replace=False)
                X_shap     = X_test_A.iloc[shap_idx]
                explainer  = shap.TreeExplainer(xgb_model)
                shap_vals  = explainer.shap_values(X_shap)

            fig1, _ = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                shap_vals, X_shap, feature_names=FEATURES_A,
                plot_type="dot", show=False, max_display=12,
            )
            plt.title("SHAP Beeswarm — XGBoost Scenario A", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

            fig2, _ = plt.subplots(figsize=(9, 5))
            shap.summary_plot(
                shap_vals, X_shap, feature_names=FEATURES_A,
                plot_type="bar", show=False, max_display=12,
            )
            plt.title("Mean |SHAP Value| — Feature Importance", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            st.caption(
                "SHAP beeswarm (top): each point is one prediction. Colour = feature value. "
                "Dept and Size drive the largest impact. Bar chart (bottom) shows mean absolute SHAP."
            )


# =============================================================================
# PAGE 6 — Interactive Forecast
# =============================================================================

elif page == "Interactive Forecast":
    st.title("Interactive Sales Forecast")
    st.markdown(
        "Enter the details below to get an instant weekly sales prediction "
        "from the trained XGBoost model (Scenario A)."
    )

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Store & Department")

        store_id = st.selectbox("Store ID", options=list(range(1, 46)), index=0)

        # Auto-fill Type and Size based on selected store
        store_type_str, store_size = STORE_INFO[store_id]
        store_type_enc = TYPE_ENCODE[store_type_str]

        st.info(
            f"Store {store_id} — Type **{store_type_str}** | "
            f"Size: **{store_size:,} sq ft**"
        )

        dept_id = st.selectbox(
            "Department ID",
            options=sorted(df["Dept"].unique().tolist()),
            index=0,
        )

        st.subheader("Week")
        forecast_date = st.date_input(
            "Forecast Date (any Friday in the week)",
            value=pd.Timestamp("2012-10-05").date(),
        )
        is_holiday = st.selectbox(
            "Is this a holiday week?",
            options=[0, 1],
            format_func=lambda x: "Yes (Holiday)" if x == 1 else "No (Regular week)",
            index=0,
        )

    with col_right:
        st.subheader("Economic Indicators")

        temperature  = st.slider("Temperature (°F)", min_value=-10.0, max_value=110.0, value=62.5, step=0.5)
        fuel_price   = st.slider("Fuel Price ($/gallon)", min_value=2.0,  max_value=5.5,  value=3.52, step=0.01)
        cpi          = st.slider("CPI (Consumer Price Index)", min_value=120.0, max_value=230.0, value=211.0, step=0.5)
        unemployment = st.slider("Unemployment Rate (%)", min_value=3.0, max_value=15.0, value=8.1, step=0.1)

    st.markdown("---")

    if st.button("Predict Weekly Sales", type="primary", use_container_width=True):
        d = pd.Timestamp(forecast_date)
        input_row = pd.DataFrame([{
            "Store":        store_id,
            "Dept":         dept_id,
            "year":         d.year,
            "month":        d.month,
            "week":         d.isocalendar()[1],
            "IsHoliday":    is_holiday,
            "Type":         store_type_enc,
            "Size":         store_size,
            "Temperature":  temperature,
            "Fuel_Price":   fuel_price,
            "CPI":          cpi,
            "Unemployment": unemployment,
        }])

        prediction = max(0.0, float(xgb_model.predict(xgb.DMatrix(input_row[FEATURES_A]))[0]))

        st.success(f"### Predicted Weekly Sales: **${prediction:,.2f}**")

        # Show context — how this compares to dataset averages
        st.markdown("---")
        st.subheader("Context")

        store_avg = df[df["Store"] == store_id]["Weekly_Sales"].mean()
        dept_avg  = df[df["Dept"]  == dept_id ]["Weekly_Sales"].mean()
        overall_avg = df["Weekly_Sales"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Your Prediction",     f"${prediction:,.0f}")
        c2.metric("Store Avg (all depts)", f"${store_avg:,.0f}")
        c3.metric("Dept Avg (all stores)", f"${dept_avg:,.0f}")
        c4.metric("Overall Avg",           f"${overall_avg:,.0f}")

        # Historical chart for this store/dept
        store_dept_df = df[(df["Store"] == store_id) & (df["Dept"] == dept_id)].copy()
        if len(store_dept_df) > 0:
            fig, ax = plt.subplots(figsize=(13, 4))
            ax.plot(
                store_dept_df["Date"], store_dept_df["Weekly_Sales"],
                color="steelblue", linewidth=1.2, label=f"Historical — Store {store_id}, Dept {dept_id}",
            )
            ax.axhline(prediction, color="orange", linestyle="--", linewidth=1.5, label="Your prediction")
            ax.set_title(f"Historical Sales — Store {store_id}, Dept {dept_id}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Weekly Sales ($)")
            ax.legend()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No historical data available for this Store/Dept combination.")

    st.markdown("---")
    st.subheader("Batch Examples")
    st.markdown("Pre-loaded example predictions:")

    examples = [
        (1,  1,  "2012-10-05", 0, 62.5, 3.52, 211.0, 8.1),
        (1,  1,  "2012-11-23", 1, 45.2, 3.47, 212.5, 8.1),
        (20, 72, "2012-12-28", 1, 38.1, 3.29, 210.2, 7.9),
    ]
    example_labels = [
        "Store 1, Dept 1 — Regular week (Oct 2012)",
        "Store 1, Dept 1 — Thanksgiving week (Nov 2012)",
        "Store 20, Dept 72 — Christmas week (Dec 2012)",
    ]

    rows = []
    for (s, d_id, date_str, hol, temp, fuel, cpi_val, unemp), label in zip(examples, example_labels):
        dt = pd.Timestamp(date_str)
        s_type_str, s_size = STORE_INFO[s]
        inp = pd.DataFrame([{
            "Store": s, "Dept": d_id,
            "year": dt.year, "month": dt.month, "week": dt.isocalendar()[1],
            "IsHoliday": hol, "Type": TYPE_ENCODE[s_type_str], "Size": s_size,
            "Temperature": temp, "Fuel_Price": fuel, "CPI": cpi_val, "Unemployment": unemp,
        }])
        pred = max(0.0, float(xgb_model.predict(xgb.DMatrix(inp[FEATURES_A]))[0]))
        rows.append({
            "Example":         label,
            "Store Type":      s_type_str,
            "Store Size (sqft)": f"{s_size:,}",
            "Holiday?":        "Yes" if hol else "No",
            "Predicted Sales": f"${pred:,.2f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# app.py
# Smarty Stock Pro — Forecasting + Fuzzy Decision Support + Anomaly Detection
# Final merged file (with consistent Store → colour mapping across ALL graphs)

import os
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score

import skfuzzy as fuzz
import skfuzzy.control as ctrl


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Smarty Stock Pro", page_icon="📦", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def scroll_to_top():
    components.html(
        """
        <script>
          const main = window.parent.document.querySelector('section.main');
          if (main) { main.scrollTo(0, 0); }
          window.scrollTo(0, 0);
        </script>
        """,
        height=0,
    )


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def validate_dataset(df: pd.DataFrame) -> list[str]:
    required = ["Units Sold"]
    return [c for c in required if c not in df.columns]


def preprocess_features(df_raw: pd.DataFrame):
    df = df_raw.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["Weekday"] = df["Date"].dt.weekday
        df_feat = df.drop(columns=["Date"])
    else:
        df_feat = df

    for c in ["Demand Forecast", "Predicted Demand", "Residual", "Abs Residual"]:
        if c in df_feat.columns:
            df_feat = df_feat.drop(columns=[c])

    y = pd.to_numeric(df_feat["Units Sold"], errors="coerce").fillna(0.0).astype(float)
    X = df_feat.drop(columns=["Units Sold"])

    inv = None
    if "Inventory Level" in df_raw.columns:
        inv = pd.to_numeric(df_raw["Inventory Level"], errors="coerce").fillna(0.0).values.astype(float)

    X = pd.get_dummies(X, drop_first=True)
    return X, y, inv


def compute_universe_max(demand_pred: np.ndarray, inventory_vals: np.ndarray | None):
    d_max = float(np.nanpercentile(demand_pred, 95))
    d_max = max(d_max, 100.0)
    d_max = float(np.ceil(d_max / 50.0) * 50.0)

    if inventory_vals is None:
        i_max = d_max
    else:
        i_max = float(np.nanpercentile(inventory_vals, 95))
        i_max = max(i_max, 100.0)
        i_max = float(np.ceil(i_max / 50.0) * 50.0)

    return d_max, i_max


def build_fuzzy_system(d_max: float, i_max: float) -> ctrl.ControlSystem:
    demand = ctrl.Antecedent(np.arange(0, int(d_max) + 1, 1), "demand")
    inventory = ctrl.Antecedent(np.arange(0, int(i_max) + 1, 1), "inventory")

    action = ctrl.Consequent(np.arange(0, 101, 1), "action")
    priority = ctrl.Consequent(np.arange(0, 101, 1), "priority")

    demand["low"] = fuzz.trimf(demand.universe, [0, 0, 0.40 * d_max])
    demand["medium"] = fuzz.trimf(demand.universe, [0.25 * d_max, 0.55 * d_max, 0.85 * d_max])
    demand["high"] = fuzz.trimf(demand.universe, [0.60 * d_max, d_max, d_max])

    inventory["low"] = fuzz.trimf(inventory.universe, [0, 0, 0.40 * i_max])
    inventory["medium"] = fuzz.trimf(inventory.universe, [0.25 * i_max, 0.55 * i_max, 0.85 * i_max])
    inventory["high"] = fuzz.trimf(inventory.universe, [0.60 * i_max, i_max, i_max])

    action["reduce"] = fuzz.trimf(action.universe, [0, 0, 40])
    action["maintain"] = fuzz.trimf(action.universe, [30, 50, 70])
    action["reorder"] = fuzz.trimf(action.universe, [60, 100, 100])

    priority["low"] = fuzz.trimf(priority.universe, [0, 0, 40])
    priority["medium"] = fuzz.trimf(priority.universe, [30, 50, 70])
    priority["high"] = fuzz.trimf(priority.universe, [60, 100, 100])

    rules = [
        ctrl.Rule(demand["high"] & inventory["low"], (action["reorder"], priority["high"])),
        ctrl.Rule(demand["high"] & inventory["medium"], (action["reorder"], priority["high"])),
        ctrl.Rule(demand["high"] & inventory["high"], (action["maintain"], priority["medium"])),

        ctrl.Rule(demand["medium"] & inventory["low"], (action["reorder"], priority["medium"])),
        ctrl.Rule(demand["medium"] & inventory["medium"], (action["maintain"], priority["medium"])),
        ctrl.Rule(demand["medium"] & inventory["high"], (action["reduce"], priority["low"])),

        ctrl.Rule(demand["low"] & inventory["low"], (action["maintain"], priority["low"])),
        ctrl.Rule(demand["low"] & inventory["medium"], (action["reduce"], priority["low"])),
        ctrl.Rule(demand["low"] & inventory["high"], (action["reduce"], priority["low"])),
    ]
    return ctrl.ControlSystem(rules)


def style_recommendations(df: pd.DataFrame):
    def row_style(row):
        pri = row.get("Priority Level", "")
        if pri == "High":
            return ["background-color: rgba(239,68,68,0.14);"] * len(row)
        if pri == "Medium":
            return ["background-color: rgba(245,158,11,0.14);"] * len(row)
        if pri == "Low":
            return ["background-color: rgba(34,197,94,0.12);"] * len(row)
        return [""] * len(row)

    return df.style.apply(row_style, axis=1).format(precision=2)


def safe_dataframe(df: pd.DataFrame, height: int, styled: bool):
    if not styled:
        st.dataframe(df, use_container_width=True, height=height)
        return

    max_cells = 250_000
    cells = int(df.shape[0] * df.shape[1])
    if cells > max_cells:
        st.dataframe(df, use_container_width=True, height=height)
    else:
        st.dataframe(style_recommendations(df), use_container_width=True, height=height)


def business_columns(df: pd.DataFrame) -> list[str]:
    wanted = [
        "Date",
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Inventory Level",
        "Units Sold",
        "Predicted Demand",
        "Residual",
        "Recommended Action",
        "Priority Level",
        "Priority Score",
        "Anomaly",
        "Anomaly Score",
        "Residual Z",
    ]
    return [c for c in wanted if c in df.columns]


def kpi_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
          </div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_filtered_view(df: pd.DataFrame,
                      store_sel: list[str],
                      prod_sel: list[str],
                      date_start,
                      date_end,
                      anom_only: bool) -> pd.DataFrame:
    out = df.copy()

    if "Store ID" in out.columns and store_sel:
        stores = [str(x) for x in store_sel]
        out = out[out["Store ID"].astype(str).isin(stores)]

    if "Product ID" in out.columns and prod_sel:
        prods = [str(x) for x in prod_sel]
        out = out[out["Product ID"].astype(str).isin(prods)]

    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        if date_start is not None and date_end is not None:
            out = out[(out["Date"] >= pd.to_datetime(date_start)) & (out["Date"] <= pd.to_datetime(date_end))]

    if anom_only and "Anomaly" in out.columns:
        out = out[out["Anomaly"] == True]

    if "Date" in out.columns and out["Date"].notna().any():
        out = out.sort_values("Date")

    return out


def _scope_text(store_sel: list[str], prod_sel: list[str]) -> str:
    def short_list(xs: list[str], limit: int = 2) -> str:
        if len(xs) <= limit:
            return ", ".join(xs)
        return ", ".join(xs[:limit]) + f" +{len(xs) - limit} more"

    s = "All Stores" if not store_sel else f"{len(store_sel)} store(s): {short_list([str(x) for x in store_sel])}"
    p = "All Products" if not prod_sel else f"{len(prod_sel)} product(s): {short_list([str(x) for x in prod_sel])}"
    return f"**{s}** · **{p}**"


# -----------------------------
# Consistent Store → Colour mapping (matplotlib default cycle)
# -----------------------------
def store_color_map(stores: list[str]) -> dict[str, str]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    stores_sorted = sorted({str(s) for s in stores})
    return {s: cycle[i % len(cycle)] for i, s in enumerate(stores_sorted)}


# -----------------------------
# Dataset loading + signatures
# -----------------------------
def _md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


@st.cache_data(show_spinner=False)
def cached_read_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def cached_read_csv_from_path(path: str, mtime: float, size: int) -> pd.DataFrame:
    return pd.read_csv(path)


def load_dataset(upload_bytes: bytes | None, upload_name: str | None, fallback_path: str):
    if upload_bytes:
        df = cached_read_csv_from_bytes(upload_bytes)
        sig = f"upload:{_md5_bytes(upload_bytes)}"
        label = upload_name or "uploaded.csv"
        return df, sig, label

    try:
        st_ = os.stat(fallback_path)
        mtime = float(st_.st_mtime)
        size = int(st_.st_size)
    except Exception:
        mtime, size = 0.0, 0

    df = cached_read_csv_from_path(fallback_path, mtime, size)
    sig = f"path:{fallback_path}|mtime:{mtime}|size:{size}"
    label = fallback_path
    return df, sig, label


# -----------------------------
# Cached pipeline (deterministic using ds_sig)
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_train_and_predict(df_raw: pd.DataFrame, ds_sig: str, n_estimators: int, split_ratio: float):
    X, y, inv = preprocess_features(df_raw)
    split_idx = int(len(X) * split_ratio)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X.iloc[:split_idx], y.iloc[:split_idx])

    y_test = y.iloc[split_idx:].values
    y_pred = model.predict(X.iloc[split_idx:])

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    df_test = df_raw.iloc[split_idx:].copy()
    inv_test = inv[split_idx:] if inv is not None else None

    return {
        "split_idx": split_idx,
        "df_test": df_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "r2": r2,
        "inv_test": inv_test,
    }


def generate_recommendations_and_anomalies(results: dict, contamination: float, z_thresh: float) -> pd.DataFrame:
    df_test = results["df_test"].copy()
    preds = np.asarray(results["y_pred"], dtype=float)
    actual = np.asarray(results["y_test"], dtype=float)
    inv = results["inv_test"]

    df_test["Predicted Demand"] = np.round(preds, 2)
    df_test["Units Sold"] = np.round(actual, 2)
    df_test["Residual"] = np.round(actual - preds, 2)
    df_test["Abs Residual"] = np.abs(df_test["Residual"].astype(float))

    feat = pd.DataFrame({
        "abs_resid": df_test["Abs Residual"].astype(float).values,
        "resid": df_test["Residual"].astype(float).values,
        "pred": df_test["Predicted Demand"].astype(float).values,
        "actual": df_test["Units Sold"].astype(float).values,
    })

    iso = IsolationForest(
        n_estimators=200,
        contamination=float(contamination),
        random_state=42,
    )
    iso.fit(feat)
    iso_pred = iso.predict(feat)
    iso_score = iso.decision_function(feat)

    inv_score = (-iso_score).astype(float)
    smin, smax = float(np.min(inv_score)), float(np.max(inv_score))
    if smax - smin < 1e-9:
        anom_score = np.zeros_like(inv_score)
    else:
        anom_score = (inv_score - smin) / (smax - smin)

    df_test["Anomaly Score"] = np.round(anom_score, 3)

    abs_res = df_test["Abs Residual"].astype(float).values
    mu = float(np.mean(abs_res))
    sd = float(np.std(abs_res)) if float(np.std(abs_res)) > 1e-9 else 1.0
    z = (abs_res - mu) / sd
    df_test["Residual Z"] = np.round(z, 2)

    df_test["Anomaly"] = (iso_pred == -1) | (df_test["Residual Z"].astype(float) >= float(z_thresh))

    # Fuzzy decision support
    if inv is None:
        df_test["Recommended Action"] = "Inventory data required"
        df_test["Priority Level"] = "N/A"
        df_test["Priority Score"] = np.nan
        return df_test

    inv = np.asarray(inv, dtype=float)
    df_test["Inventory Level"] = np.round(inv, 2)

    d_max, i_max = compute_universe_max(preds, inv)
    fuzzy_system = build_fuzzy_system(d_max, i_max)

    actions, priorities, p_scores = [], [], []
    for d_hat, inv_val in zip(preds, inv):
        sim = ctrl.ControlSystemSimulation(fuzzy_system)
        sim.input["demand"] = float(np.clip(d_hat, 0, d_max))
        sim.input["inventory"] = float(np.clip(inv_val, 0, i_max))
        sim.compute()

        action_score = float(sim.output.get("action", 50))
        priority_score = float(sim.output.get("priority", 50))

        if action_score < 33:
            action_label = "Reduce excess stock"
        elif action_score < 66:
            action_label = "Maintain current level"
        else:
            action_label = "Restock inventory"

        if priority_score < 33:
            priority_label = "Low"
        elif priority_score < 66:
            priority_label = "Medium"
        else:
            priority_label = "High"

        actions.append(action_label)
        priorities.append(priority_label)
        p_scores.append(round(priority_score, 2))

    df_test["Recommended Action"] = actions
    df_test["Priority Level"] = priorities
    df_test["Priority Score"] = p_scores
    return df_test


@st.cache_data(show_spinner=False)
def cached_recos_and_anoms(df_test: pd.DataFrame,
                           ds_sig: str,
                           y_test: np.ndarray,
                           y_pred: np.ndarray,
                           inv_test,
                           contamination: float,
                           z_thresh: float) -> pd.DataFrame:
    results = {"df_test": df_test, "y_test": y_test, "y_pred": y_pred, "inv_test": inv_test}
    return generate_recommendations_and_anomalies(results, contamination=contamination, z_thresh=z_thresh)


# -----------------------------
# Panels
# -----------------------------
def whats_happening_panel(df_view: pd.DataFrame, store_sel: list[str], prod_sel: list[str]):
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("### What’s happening")
    st.markdown(_scope_text(store_sel, prod_sel))

    if df_view is None or len(df_view) == 0:
        st.info("No rows match the current drilldown. Try widening the date range or selecting fewer filters.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if "Date" in df_view.columns and df_view["Date"].notna().any():
        latest = df_view.dropna(subset=["Date"]).iloc[-1]
        latest_date = latest.get("Date", None)
        date_str = latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else ""
    else:
        latest = df_view.iloc[-1]
        date_str = ""

    latest_actual = float(latest.get("Units Sold", np.nan))
    latest_pred = float(latest.get("Predicted Demand", np.nan))
    latest_resid = float(latest.get("Residual", np.nan))
    latest_anom = bool(latest.get("Anomaly", False))
    latest_pri = str(latest.get("Priority Level", "N/A"))
    latest_action = str(latest.get("Recommended Action", "N/A"))

    st.markdown(
        f"- Latest {f'({date_str})' if date_str else ''}: "
        f"Actual **{latest_actual:.2f}**, Predicted **{latest_pred:.2f}**, Residual **{latest_resid:.2f}**"
    )
    st.markdown(f"- Priority: **{latest_pri}** · Action: **{latest_action}** · Anomaly: **{'Yes' if latest_anom else 'No'}**")
    st.markdown("</div>", unsafe_allow_html=True)


def recommended_actions_panel(df_view: pd.DataFrame, contamination: float, z_thresh: float):
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("### Recommended actions")

    if df_view is None or len(df_view) == 0:
        st.info("No recommendations to show for the current drilldown.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    anom_n = int(df_view["Anomaly"].sum()) if "Anomaly" in df_view.columns else 0
    restock_n = int((df_view.get("Recommended Action", "") == "Restock inventory").sum())
    reduce_n = int((df_view.get("Recommended Action", "") == "Reduce excess stock").sum())

    bullets = []
    if restock_n > 0:
        bullets.append("Restock the highest-priority SKUs first; validate lead time and allocate inventory by store demand.")
    if reduce_n > 0:
        bullets.append("For overstock items, consider transfer to higher-demand stores, markdown, or a purchasing pause.")
    if anom_n > 0:
        bullets.append("Investigate anomalies (unexpected spike/drop vs forecast): check promos, pricing, stockouts, and data gaps.")
    bullets.append("Use the Act page tables to download the filtered list for execution.")

    for b in bullets[:6]:
        st.markdown(f"- {b}")

    with st.expander("Show reasoning (metrics)"):
        resid = df_view["Residual"].astype(float) if "Residual" in df_view.columns else pd.Series([], dtype=float)
        mean_resid = float(resid.mean()) if len(resid) else 0.0
        std_resid = float(resid.std(ddof=0)) if len(resid) else 0.0
        st.markdown(
            f"- Mean residual: **{mean_resid:.2f}** (Actual − Predicted)\n"
            f"- Residual std: **{std_resid:.2f}**\n"
            f"- Anomalies in view: **{anom_n}**\n"
            f"- IsolationForest contamination: **{contamination:.3f}**\n"
            f"- Residual Z threshold: **{z_thresh:.1f}**"
        )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Plots (ALL use Store → colour mapping)
# -----------------------------
def plot_scatter_actual_vs_pred_multi(df_view: pd.DataFrame, title: str = "Actual vs Predicted"):
    dfp = df_view.copy()
    needed = ["Units Sold", "Predicted Demand"]
    if not all(c in dfp.columns for c in needed):
        fig = plt.figure(figsize=(7, 5))
        plt.title(title)
        plt.text(0.5, 0.5, "Units Sold / Predicted Demand not available.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        return fig

    dfp["Units Sold"] = pd.to_numeric(dfp["Units Sold"], errors="coerce")
    dfp["Predicted Demand"] = pd.to_numeric(dfp["Predicted Demand"], errors="coerce")
    dfp = dfp.dropna(subset=["Units Sold", "Predicted Demand"])

    fig = plt.figure(figsize=(7, 5))

    if "Store ID" in dfp.columns and dfp["Store ID"].notna().any():
        stores = dfp["Store ID"].astype(str).tolist()
        cmap = store_color_map(stores)
        for store, g in dfp.groupby(dfp["Store ID"].astype(str)):
            plt.scatter(
                g["Units Sold"],
                g["Predicted Demand"],
                alpha=0.35,
                label=f"Store {store}",
                color=cmap.get(str(store), None),
            )
    else:
        plt.scatter(dfp["Units Sold"], dfp["Predicted Demand"], alpha=0.35, label="All")

    mn = float(min(dfp["Units Sold"].min(), dfp["Predicted Demand"].min()))
    mx = float(max(dfp["Units Sold"].max(), dfp["Predicted Demand"].max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")

    plt.xlabel("Actual Units Sold")
    plt.ylabel("Predicted Units Sold")
    plt.title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    if len(labels) > 1:
        plt.legend(loc="best", frameon=True)

    plt.tight_layout()
    return fig


def plot_residual_hist_multi(df_view: pd.DataFrame, title: str = "Residual distribution", split_by_store: bool = True):
    dfp = df_view.copy()

    if "Residual" not in dfp.columns:
        fig = plt.figure(figsize=(7, 5))
        plt.title(title)
        plt.text(0.5, 0.5, "Residual not available.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        return fig

    dfp["Residual"] = pd.to_numeric(dfp["Residual"], errors="coerce")
    dfp = dfp.dropna(subset=["Residual"])

    fig = plt.figure(figsize=(7, 5))

    if split_by_store and "Store ID" in dfp.columns and dfp["Store ID"].notna().any():
        stores = dfp["Store ID"].astype(str).tolist()
        cmap = store_color_map(stores)
        for store, g in dfp.groupby(dfp["Store ID"].astype(str)):
            plt.hist(
                g["Residual"].values,
                bins=35,
                alpha=0.30,
                label=f"Store {store}",
                color=cmap.get(str(store), None),
            )
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(labels) > 1:
            plt.legend(loc="best", frameon=True)
    else:
        plt.hist(dfp["Residual"].values, bins=40, alpha=0.85, label="Residual")
        plt.legend(loc="best", frameon=True)

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Residual (Actual − Predicted)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_residuals_over_time_multi(df_plot: pd.DataFrame, title: str):
    dfp = df_plot.copy()

    if "Date" not in dfp.columns or "Residual" not in dfp.columns:
        fig = plt.figure(figsize=(10, 4.7))
        plt.title(title)
        plt.text(0.5, 0.5, "Date/Residual not available for plotting.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        return fig

    dfp["Date"] = pd.to_datetime(dfp["Date"], errors="coerce")
    dfp["Residual"] = pd.to_numeric(dfp["Residual"], errors="coerce")
    dfp = dfp.dropna(subset=["Date", "Residual"])

    if len(dfp) == 0:
        fig = plt.figure(figsize=(10, 4.7))
        plt.title(title)
        plt.text(0.5, 0.5, "No valid dates in the selected scope.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        return fig

    has_store = "Store ID" in dfp.columns and dfp["Store ID"].notna().any()
    has_prod = "Product ID" in dfp.columns and dfp["Product ID"].notna().any()

    fig = plt.figure(figsize=(10, 4.7))

    if has_store:
        cmap = store_color_map(dfp["Store ID"].astype(str).tolist())
    else:
        cmap = {}

    # When multiple products exist, keep store colour consistent, differentiate products by linestyle
    linestyles = ["-", "--", ":", "-."]

    if has_store and has_prod:
        grouped = dfp.groupby([dfp["Store ID"].astype(str), dfp["Product ID"].astype(str)], dropna=False)
        # stable order for legend
        keys_sorted = sorted(list(grouped.groups.keys()), key=lambda x: (x[0], x[1]))
        for (store, prod) in keys_sorted:
            g = grouped.get_group((store, prod)).sort_values("Date")
            # choose linestyle based on product index within store
            prod_list = sorted(dfp[dfp["Store ID"].astype(str) == store]["Product ID"].astype(str).unique().tolist())
            ls = linestyles[prod_list.index(prod) % len(linestyles)] if prod in prod_list else "-"
            plt.plot(
                g["Date"],
                g["Residual"],
                marker="o",
                linestyle=ls,
                linewidth=1,
                label=f"Store {store} · Product {prod}",
                color=cmap.get(store, None),
            )
            if "Anomaly" in g.columns and g["Anomaly"].any():
                an = g[g["Anomaly"] == True]
                plt.scatter(
                    an["Date"],
                    an["Residual"],
                    marker="x",
                    s=70,
                    color=cmap.get(store, None),
                )

    elif has_store:
        grouped = dfp.groupby(dfp["Store ID"].astype(str), dropna=False)
        stores_sorted = sorted(list(grouped.groups.keys()))
        for store in stores_sorted:
            g = grouped.get_group(store).sort_values("Date")
            plt.plot(
                g["Date"],
                g["Residual"],
                marker="o",
                linestyle="-",
                linewidth=1,
                label=f"Store {store}",
                color=cmap.get(store, None),
            )
            if "Anomaly" in g.columns and g["Anomaly"].any():
                an = g[g["Anomaly"] == True]
                plt.scatter(
                    an["Date"],
                    an["Residual"],
                    marker="x",
                    s=70,
                    color=cmap.get(store, None),
                )
    else:
        dfp = dfp.sort_values("Date")
        plt.plot(dfp["Date"], dfp["Residual"], marker="o", linestyle="-", linewidth=1, label="Residual")
        if "Anomaly" in dfp.columns and dfp["Anomaly"].any():
            an = dfp[dfp["Anomaly"] == True]
            plt.scatter(an["Date"], an["Residual"], marker="x", s=70, label="Anomaly")

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Residual (Actual − Predicted)")
    plt.title(title)
    plt.xticks(rotation=20)

    handles, labels = plt.gca().get_legend_handles_labels()
    if len(labels) > 1:
        plt.legend(loc="best", frameon=True)

    plt.tight_layout()
    return fig


# -----------------------------
# CSS (brand + usability)
# -----------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.05rem; padding-bottom: 2rem; }
      .stApp { background: #f6f7fb; }

      /* Sidebar: brand */
      section[data-testid="stSidebar"] {
        background: #144a42 !important;
        border-right: 1px solid rgba(255,255,255,0.10);
      }
      section[data-testid="stSidebar"] * { color: #ffffff !important; }
      section[data-testid="stSidebar"] .stCaption,
      section[data-testid="stSidebar"] small,
      section[data-testid="stSidebar"] p { color: rgba(255,255,255,0.80) !important; }

      /* Sidebar buttons like menu items */
      section[data-testid="stSidebar"] .stButton>button {
        background: transparent !important;
        border: 0 !important;
        text-align: left !important;
        padding: 14px 14px !important;
        border-radius: 16px !important;
        font-size: 18px !important;
        font-weight: 760 !important;
        width: 100% !important;
      }
      section[data-testid="stSidebar"] .stButton>button:hover {
        background: rgba(255,255,255,0.10) !important;
      }

      .nav-section {
        margin: 18px 6px 10px 6px;
        font-size: 12px;
        font-weight: 780;
        letter-spacing: 1.4px;
        opacity: .65;
        text-transform: uppercase;
      }

      /* Sidebar uploader readable */
      section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 14px !important;
      }
      section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button{
        background: #ffffff !important;
        color: #0f172a !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
      }
      section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button *{
        color: #0f172a !important;
      }

      /* KPI cards */
      .kpi-card {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 10px 28px rgba(17, 24, 39, 0.07);
      }
      .kpi-label { font-size: 0.90rem; color: rgba(17,24,39,0.68); font-weight: 650; }
      .kpi-value { font-size: 2.05rem; font-weight: 820; color: rgba(17,24,39,0.94); line-height: 1.05; margin-top: 4px; }
      .kpi-sub   { font-size: 0.86rem; color: rgba(17,24,39,0.62); margin-top: 6px; }

      .dataframe-wrap {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 16px;
        padding: 12px;
        box-shadow: 0 8px 24px rgba(17, 24, 39, 0.05);
      }

      .panel-card {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 10px 28px rgba(17, 24, 39, 0.06);
      }

      /* Primary buttons */
      .stButton>button[kind="primary"] {
        background: linear-gradient(180deg, #2563eb, #1d4ed8) !important;
        border: 1px solid rgba(0,0,0,0.04) !important;
        color: #fff !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        padding: 0.65rem 0.9rem !important;
      }

      /* Remove the long rounded white expander header bar */
      details summary {
        background: transparent !important;
        border: none !important;
        border-radius: 0px !important;
        padding: 0.35rem 0 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("")
st.title("📦 Smarty Stock Pro")
st.caption("Demand forecasting, decision support, and anomaly monitoring for retail inventory.")
st.divider()


# -----------------------------
# Session state defaults
# -----------------------------
DEFAULT_DATA_PATH = "retail_store_inventory.csv"

defaults = {
    "nav_page": "Overview",

    # settings
    "trees": 140,
    "split_pct": 80,
    "contamination": 0.02,
    "z_thresh": 3.0,

    # upload stored as bytes
    "upload_bytes": None,
    "upload_name": None,

    # pipeline
    "results": None,
    "results_table": None,
    "trained": False,
    "last_run_time": None,
    "dataset_sig": None,
    "settings_sig": None,

    # drilldown
    "store_sel": [],
    "prod_sel": [],
    "anom_only": False,
    "date_start": None,
    "date_end": None,

    "_init_done": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -----------------------------
# Sidebar menu (buttons)
# -----------------------------
def nav_btn(label: str, key: str, target: str):
    active = (st.session_state["nav_page"] == target)
    if st.sidebar.button(label, key=key, use_container_width=True, type=("primary" if active else "secondary")):
        st.session_state["nav_page"] = target
        scroll_to_top()


with st.sidebar:
    st.markdown('<div class="nav-section">Dashboards</div>', unsafe_allow_html=True)
    nav_btn("Overview", "nav_overview", "Overview")
    nav_btn("⌁  Investigate", "nav_investigate", "Investigate")

    st.markdown('<div class="nav-section">Operations</div>', unsafe_allow_html=True)
    nav_btn("✅  Act", "nav_act", "Act")

    st.markdown('<div class="nav-section">Settings</div>', unsafe_allow_html=True)
    nav_btn("⚙️  Settings", "nav_settings", "Settings")

page = st.session_state["nav_page"]


# -----------------------------
# Run logic
# -----------------------------
def settings_sig():
    return (
        int(st.session_state["trees"]),
        int(st.session_state["split_pct"]),
        float(st.session_state["contamination"]),
        float(st.session_state["z_thresh"]),
    )


def do_reset():
    keep_page = st.session_state.get("nav_page", "Overview")
    st.session_state.clear()
    for k, v in defaults.items():
        st.session_state[k] = v
    st.session_state["nav_page"] = keep_page
    st.rerun()


def run_analysis(force: bool = False):
    df_raw, ds_sig, ds_label = load_dataset(
        st.session_state.get("upload_bytes"),
        st.session_state.get("upload_name"),
        DEFAULT_DATA_PATH,
    )

    missing = validate_dataset(df_raw)
    if missing:
        st.error(f"Dataset must contain: {', '.join(missing)}")
        return False

    df_raw = ensure_datetime(df_raw)

    prev_ds_sig = st.session_state.get("dataset_sig")
    new_settings_sig = settings_sig()

    needs_run = (
        force
        or st.session_state.get("results") is None
        or st.session_state.get("results_table") is None
        or prev_ds_sig != ds_sig
    )

    if not needs_run:
        return True

    trees = int(st.session_state["trees"])
    split_ratio = float(st.session_state["split_pct"]) / 100.0
    contamination = float(st.session_state["contamination"])
    zt = float(st.session_state["z_thresh"])

    with st.spinner("Running forecasting → anomaly detection → decision support..."):
        res = cached_train_and_predict(df_raw, ds_sig, trees, split_ratio)
        tab = cached_recos_and_anoms(
            res["df_test"],
            ds_sig,
            res["y_test"],
            res["y_pred"],
            res["inv_test"],
            contamination,
            zt,
        )

        st.session_state["results"] = {
            **res,
            "dataset_label": ds_label,
            "meta": {"trees": trees, "split": split_ratio},
        }
        st.session_state["results_table"] = tab
        st.session_state["trained"] = True
        st.session_state["last_run_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["dataset_sig"] = ds_sig
        st.session_state["settings_sig"] = new_settings_sig

    return True


# Auto-run on open + dataset change
run_analysis(force=False)


# Settings-changed banner
current_sig = settings_sig()
applied_sig = st.session_state.get("settings_sig")
if applied_sig is not None and current_sig != applied_sig:
    st.warning("Settings changed — click **Run analysis** in **Settings** to apply.", icon="⚠️")


# -----------------------------
# Settings page
# -----------------------------
if page == "Settings":
    st.subheader("Settings")
    st.caption("Adjust model and anomaly sensitivity, then click Run analysis to apply.")

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.markdown("### Dataset")
        up = st.file_uploader("Optional: Upload CSV", type=["csv"])
        if up is not None:
            st.session_state["upload_bytes"] = up.getvalue()
            st.session_state["upload_name"] = up.name
            run_analysis(force=False)

        st.caption("If no upload is provided, the default dataset in the project folder is used.")

        st.markdown("### Forecasting model")
        st.session_state["trees"] = st.slider("Model trees", 50, 300, int(st.session_state["trees"]), step=10)
        st.session_state["split_pct"] = st.slider("Train split (%)", 60, 90, int(st.session_state["split_pct"]), step=5)

    with right:
        st.markdown("### Anomaly detection")
        st.session_state["contamination"] = st.slider(
            "Sensitivity (contamination)", 0.005, 0.05, float(st.session_state["contamination"]), step=0.005
        )
        st.session_state["z_thresh"] = st.slider(
            "Residual Z threshold", 2.0, 5.0, float(st.session_state["z_thresh"]), step=0.5
        )
        st.caption("Isolation Forest flags unusual forecast errors; Z-score acts as a simple cross-check.")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            if st.button("Run analysis", type="primary", use_container_width=True):
                ok = run_analysis(force=True)
                if ok:
                    st.success("Updated. Open Overview / Investigate / Act.")
        with c2:
            if st.button("Reset", use_container_width=True):
                do_reset()

    st.divider()
    st.caption(f"Last run: {st.session_state.get('last_run_time') or '—'}")
    st.stop()


# -----------------------------
# Pages require results
# -----------------------------
if st.session_state.get("results") is None or st.session_state.get("results_table") is None:
    st.info("Open **Settings** and click **Run analysis**.")
    st.stop()

results = st.session_state["results"]
results_table = st.session_state["results_table"]

df_raw, _, dataset_label = load_dataset(
    st.session_state.get("upload_bytes"),
    st.session_state.get("upload_name"),
    DEFAULT_DATA_PATH,
)
df_raw = ensure_datetime(df_raw)

split_ratio = float(st.session_state["split_pct"]) / 100.0
contamination = float(st.session_state["contamination"])
z_thresh = float(st.session_state["z_thresh"])


# -----------------------------
# Drilldown (main page)
# -----------------------------
st.subheader("Drilldown")

dfp = results_table.copy()
if "Date" in dfp.columns:
    dfp["Date"] = pd.to_datetime(dfp["Date"], errors="coerce")

store_values = sorted(dfp["Store ID"].dropna().astype(str).unique().tolist()) if "Store ID" in dfp.columns else []
prod_values_all = sorted(dfp["Product ID"].dropna().astype(str).unique().tolist()) if "Product ID" in dfp.columns else []

# Smart defaults: on first open, pick most common store and product in that store (immediate "wow")
if (not st.session_state.get("_init_done")) and store_values:
    try:
        top_store = dfp["Store ID"].dropna().astype(str).value_counts().idxmax()
        st.session_state["store_sel"] = [str(top_store)]
        if "Product ID" in dfp.columns:
            top_prod = (
                dfp[dfp["Store ID"].astype(str) == str(top_store)]["Product ID"]
                .dropna().astype(str).value_counts().idxmax()
            )
            st.session_state["prod_sel"] = [str(top_prod)]
    except Exception:
        st.session_state["store_sel"] = []
        st.session_state["prod_sel"] = []
    st.session_state["_init_done"] = True

d1, d2, d3, d4, d5 = st.columns([1.35, 1.55, 1.85, 1.05, 0.9], gap="large")

with d1:
    store_sel = st.multiselect(
        "Stores",
        options=store_values,
        default=[str(x) for x in (st.session_state.get("store_sel") or []) if str(x) in store_values],
        help="Leave empty to include all stores.",
    )

with d2:
    if "Product ID" in dfp.columns:
        if store_sel and "Store ID" in dfp.columns:
            prod_values = sorted(
                dfp[dfp["Store ID"].astype(str).isin([str(x) for x in store_sel])]["Product ID"]
                .dropna().astype(str).unique().tolist()
            )
        else:
            prod_values = prod_values_all

        current_default = [str(x) for x in (st.session_state.get("prod_sel") or []) if str(x) in prod_values]
        prod_sel = st.multiselect(
            "Products",
            options=prod_values,
            default=current_default,
            help="Leave empty to include all products.",
        )
    else:
        prod_sel = []

with d3:
    date_start = None
    date_end = None
    if "Date" in dfp.columns and dfp["Date"].notna().any():
        dmin = dfp["Date"].min().date()
        dmax = dfp["Date"].max().date()

        default_start = st.session_state.get("date_start") or dmin
        default_end = st.session_state.get("date_end") or dmax

        default_start = max(dmin, min(default_start, dmax))
        default_end = max(dmin, min(default_end, dmax))
        if default_start > default_end:
            default_start, default_end = dmin, dmax

        date_range = st.date_input(
            "Date range",
            value=(default_start, default_end),
            min_value=dmin,
            max_value=dmax,
        )
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            date_start, date_end = date_range[0], date_range[1]
    else:
        st.caption("Date range unavailable (no valid Date values).")

with d4:
    anom_only = st.toggle("Anomalies only", value=bool(st.session_state.get("anom_only", False)))

with d5:
    st.markdown("**Quick**")
    if st.button("Clear", use_container_width=True):
        st.session_state["store_sel"] = []
        st.session_state["prod_sel"] = []
        st.session_state["anom_only"] = False
        st.rerun()

st.session_state["store_sel"] = store_sel
st.session_state["prod_sel"] = prod_sel
st.session_state["anom_only"] = anom_only
if date_start is not None and date_end is not None:
    st.session_state["date_start"] = date_start
    st.session_state["date_end"] = date_end

st.divider()

df_view = get_filtered_view(
    results_table,
    store_sel=store_sel,
    prod_sel=prod_sel,
    date_start=date_start,
    date_end=date_end,
    anom_only=anom_only,
)

# KPIs
total_rows = len(df_raw)
test_rows = len(results["y_pred"])
holdout_pct = int((1 - split_ratio) * 100)

high_n_view = int((df_view.get("Priority Level", "") == "High").sum()) if len(df_view) else 0
anom_n_view = int(df_view.get("Anomaly", pd.Series([], dtype=bool)).sum()) if len(df_view) else 0
restock_n_view = int((df_view.get("Recommended Action", "") == "Restock inventory").sum()) if len(df_view) else 0


# -----------------------------
# Pages
# -----------------------------
if page == "Overview":
    a, b, c, d = st.columns(4, gap="large")
    with a: kpi_card("Dataset rows", f"{total_rows:,}", f"Source: {os.path.basename(dataset_label)}")
    with b: kpi_card("Evaluation rows", f"{test_rows:,}", f"Hold-out: {holdout_pct}%")
    with c: kpi_card("MAE", f"{results['mae']:.2f}", "Avg units off per prediction")
    with d: kpi_card("R²", f"{results['r2']:.3f}", "Explained demand variability")

    st.divider()

    left, right = st.columns([1.45, 1], gap="large")

    with left:
        st.subheader("Key trend")
        if "Date" in df_view.columns and df_view["Date"].notna().any() and "Residual" in df_view.columns:
            df_plot = df_view.dropna(subset=["Date"]).copy()
            st.pyplot(plot_residuals_over_time_multi(df_plot, "Residuals over time (Actual − Predicted)"))
        else:
            # fallback if Date missing: show scatter in a consistent way
            st.pyplot(plot_scatter_actual_vs_pred_multi(df_view, "Actual vs Predicted (selected scope)"))

    with right:
        st.subheader("Decision panel")
        whats_happening_panel(df_view, store_sel, prod_sel)

        r1, r2 = st.columns(2, gap="large")
        with r1: kpi_card("High priority (view)", f"{high_n_view:,}", "Filtered scope")
        with r2: kpi_card("Anomalies (view)", f"{anom_n_view:,}", "Filtered scope")

    #st.divider()
    recommended_actions_panel(df_view, contamination, z_thresh)

    st.divider()

    tab1, tab2 = st.tabs(["High Priority (Top 50)", "Top 25 Anomalies"])

    with tab1:
        watch = df_view.copy()
        if "Priority Level" in watch.columns:
            watch = watch[watch["Priority Level"] == "High"]
        if "Priority Score" in watch.columns:
            watch = watch.sort_values("Priority Score", ascending=False)
        watch = watch[business_columns(watch)].head(50)

        if len(watch) == 0:
            st.info("No items flagged as High priority in the current view.")
        else:
            st.markdown("<div class='dataframe-wrap'>", unsafe_allow_html=True)
            safe_dataframe(watch, height=420, styled=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        anom = df_view.copy()
        if "Anomaly" in anom.columns:
            anom = anom[anom["Anomaly"] == True]
        if "Anomaly Score" in anom.columns:
            anom = anom.sort_values("Anomaly Score", ascending=False)
        anom = anom[business_columns(anom)].head(25)

        if len(anom) == 0:
            st.info("No anomalies detected in the current view.")
        else:
            st.markdown("<div class='dataframe-wrap'>", unsafe_allow_html=True)
            safe_dataframe(anom, height=420, styled=False)
            st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        "Download full recommendations (CSV)",
        data=results_table.to_csv(index=False),
        file_name="smarty_stock_pro_recommendations.csv",
        mime="text/csv",
        use_container_width=True,
    )

elif page == "Investigate":
    k1, k2, k3, k4 = st.columns(4, gap="large")
    with k1: kpi_card("Filtered rows", f"{len(df_view):,}", "After drilldown")
    with k2: kpi_card("High priority (view)", f"{high_n_view:,}", "Urgent items")
    with k3: kpi_card("Restock (view)", f"{restock_n_view:,}", "Replenishment")
    with k4: kpi_card("Anomalies (view)", f"{anom_n_view:,}", "Monitor")

    st.divider()

    st.subheader("Forecast fit (Actual vs Predicted)")
    st.pyplot(plot_scatter_actual_vs_pred_multi(df_view, "Actual vs Predicted (selected scope)"))

    st.divider()

    st.subheader("Residual distribution")
    split_store = st.toggle("Split histogram by store", value=True)
    st.pyplot(plot_residual_hist_multi(df_view, "Residual distribution (selected scope)", split_by_store=split_store))

    st.divider()

    st.subheader("Residuals over time (selected scope)")
    if "Date" in df_view.columns and df_view["Date"].notna().any() and "Residual" in df_view.columns:
        df_plot = df_view.dropna(subset=["Date"]).copy()
        st.pyplot(plot_residuals_over_time_multi(df_plot, "Residuals over time (selected scope)"))
    else:
        st.info("Time-series view requires a valid Date column in the current view.")

elif page == "Act":
    k1, k2, k3, k4 = st.columns(4, gap="large")
    with k1: kpi_card("Filtered rows", f"{len(df_view):,}", "After drilldown")
    with k2: kpi_card("High priority (view)", f"{high_n_view:,}", "Urgent items")
    with k3: kpi_card("Restock (view)", f"{restock_n_view:,}", "Replenishment")
    with k4: kpi_card("Anomalies (view)", f"{anom_n_view:,}", "Monitor")

    st.divider()

    st.subheader("Priority table")
    show_actionable_only = st.toggle("Show only actionable (High/Medium or Restock)", value=True)

    df_f = df_view.copy()
    if show_actionable_only and "Priority Level" in df_f.columns:
        mask_pri = df_f["Priority Level"].isin(["High", "Medium"])
        mask_action = (df_f.get("Recommended Action", "") == "Restock inventory")
        df_f = df_f[mask_pri | mask_action]

    if "Priority Level" in df_f.columns and "Priority Score" in df_f.columns:
        order = {"High": 0, "Medium": 1, "Low": 2}
        df_f["_rank"] = df_f["Priority Level"].map(order).fillna(9).astype(int)
        df_f = df_f.sort_values(["_rank", "Priority Score"], ascending=[True, False]).drop(columns=["_rank"])

    show_cols = business_columns(df_f)
    st.markdown("<div class='dataframe-wrap'>", unsafe_allow_html=True)
    safe_dataframe(df_f[show_cols].head(1000), height=560, styled=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.subheader("Anomalies (Top 25 by anomaly score)")
    anom = df_view.copy()
    if "Anomaly" in anom.columns:
        anom = anom[anom["Anomaly"] == True]
    if "Anomaly Score" in anom.columns:
        anom = anom.sort_values("Anomaly Score", ascending=False)

    st.markdown("<div class='dataframe-wrap'>", unsafe_allow_html=True)
    safe_dataframe(anom[show_cols].head(25), height=420, styled=False)
    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        "Download filtered view (CSV)",
        data=df_view.to_csv(index=False),
        file_name="smarty_stock_pro_filtered_view.csv",
        mime="text/csv",
        use_container_width=True,
    )


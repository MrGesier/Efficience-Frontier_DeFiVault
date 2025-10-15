import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime
import time

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="DeFi Vaults Efficient Frontier", layout="wide")
st.title("📈 DeFi Vaults Efficient Frontier")
st.caption("by **Mr_Gesier | Nomiks**")
st.markdown(
    "Visualize and optimize **DeFi yield vaults** with a Markowitz efficient frontier. "
    "Hybrid metrics (historical when available, snapshot otherwise) from "
    "[DefiLlama](https://defillama.com). Diversity-aware covariance and "
    "cardinality-constrained optimal allocation."
)

API_POOLS = "https://yields.llama.fi/pools"
API_CHART = "https://yields.llama.fi/chart/"
MAX_VAULTS = 80  # try many; filters + universe selector will narrow

# =========================================================
# DATA LAYER
# =========================================================
@st.cache_data(show_spinner=True)
def fetch_pools(limit: int = MAX_VAULTS) -> pd.DataFrame:
    r = requests.get(API_POOLS, timeout=15)
    r.raise_for_status()
    raw = r.json().get("data", [])
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    keep = ["pool","project","chain","symbol","tvlUsd","apy","apyMean30d","stablecoin","ilRisk"]
    df = df[keep].copy()
    df = df.sort_values("tvlUsd", ascending=False).head(limit)
    df["apy"] = pd.to_numeric(df["apy"], errors="coerce") / 100
    df["apyMean30d"] = pd.to_numeric(df["apyMean30d"], errors="coerce") / 100
    df["name"] = df.apply(lambda r: f"{r['project']}:{r['symbol']} ({r['chain']})", axis=1)
    df = df.dropna(subset=["apy"]).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def fetch_history(pool_id: str) -> pd.DataFrame | None:
    """Safe fetch for /chart/{pool_id} with mixed timestamp formats."""
    url = f"{API_CHART}{pool_id}"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            return None
        js = r.json()
        rows = js.get("data", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])
        df["apy"] = pd.to_numeric(df["apy"], errors="coerce") / 100
        df = df.dropna(subset=["apy"])
        return df if len(df) >= 3 else None
    except Exception:
        return None

@st.cache_data(show_spinner=True)
def build_hybrid_metrics(snapshot_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows, skipped = [], []
    progress = st.progress(0.0)
    n = len(snapshot_df)
    for i, row in snapshot_df.iterrows():
        pid = row["pool"]
        hist = fetch_history(pid)
        if hist is not None:
            mean_ret = float(hist["apy"].mean())
            vol = float(hist["apy"].std())
            n_obs = int(len(hist))
        else:
            # fallback to snapshot APY & a small vol proxy using 30d drift
            mean_ret = float(row["apy"])
            if pd.isna(row["apyMean30d"]):
                vol = 0.01
            else:
                vol = abs(float(row["apy"]) - float(row["apyMean30d"]))
                if vol <= 0:
                    vol = 0.005
            n_obs = 0
            skipped.append(pid)

        rows.append({
            "pool": pid,
            "name": row["name"],
            "project": row["project"],
            "chain": row["chain"],
            "symbol": row["symbol"],
            "tvlUsd": float(row["tvlUsd"]),
            "stablecoin": bool(row["stablecoin"]),
            "ilRisk": row["ilRisk"],
            "mean_return": max(mean_ret, -1.0),            # sanity clamps
            "volatility": max(vol, 0.0001),
            "observations": n_obs
        })
        progress.progress((i+1)/n)
        time.sleep(0.03)
    df = pd.DataFrame(rows)
    return df, skipped

# =========================================================
# CLASSIFICATION / TAXONOMY
# =========================================================
def classify(row) -> str:
    if row["stablecoin"]:
        return "Stablecoin"
    s = str(row["symbol"]).upper()
    if "BTC" in s:
        return "BTC"
    if any(k in s for k in ["ETH","STETH","WSTETH","WBETH","RETH"]):
        return "ETH / LSD"
    if row["ilRisk"] == "yes":
        return "LP / Farming"
    return "Structured / Credit"

COLOR_MAP = {
    "Stablecoin": "blue",
    "BTC": "orange",
    "ETH / LSD": "green",
    "LP / Farming": "gold",
    "Structured / Credit": "purple",
}

# =========================================================
# FRONT-END CONTROLS
# =========================================================
snap = fetch_pools(MAX_VAULTS)
st.info("⏳ Building metrics (historical when available, snapshot fallback)…")
data, skipped_ids = build_hybrid_metrics(snap)
data["category"] = data.apply(classify, axis=1)
data["color"] = data["category"].map(COLOR_MAP)

st.success(f"Loaded {len(data)} vaults — {len(data)-len(skipped_ids)} with history, {len(skipped_ids)} snapshot only.")

with st.sidebar:
    st.header("⚙️ Interactive Controls")
    # Category filter
    cats = sorted(data["category"].unique().tolist())
    selected_cats = st.multiselect(
        "Filter by Category",
        options=cats,
        default=cats,
        help="Select categories to include in your universe."
    )
    uni = data[data["category"].isin(selected_cats)].copy()

    # Explicit universe selector
    st.markdown("**Vault Universe**")
    chosen = st.multiselect(
        "Include specific vaults (leave empty = all in filtered categories):",
        options=uni["name"].tolist(),
        default=[],
        help="Pick the vaults to include in the optimizer universe."
    )
    if chosen:
        uni = uni[uni["name"].isin(chosen)].copy()

    target_return_pct = st.slider(
        "Target Return (APY %)",
        0.0, float(max(0.5, uni["mean_return"].max()*100)), 10.0,
        help="Desired annualized return for the optimal portfolio (frontier point)."
    )
    diversity_pref = st.slider(
        "Diversity Preference", 0.0, 1.0, 0.5,
        help="Higher = stronger penalization of correlated risk (promotes variety)."
    )
    max_assets = st.slider(
        "Max Assets in Portfolio", 3, min(25, max(3, len(uni))), 8,
        help="Cardinality cap. We keep top-K weights and re-normalize."
    )
    total_invest = st.number_input(
        "Total Investment (USD)", min_value=1000, max_value=10_000_000, value=50_000, step=1000,
        help="Capital to allocate across selected vaults."
    )

# =========================================================
# DIVERSITY-AWARE COVARIANCE
# =========================================================
# Base volatility vector / mean returns
mu = uni["mean_return"].values
sig = uni["volatility"].values
base_cov = np.diag(sig**2)

# Category-aware correlation (simulate intra=0.7, inter=0.2)
cats_arr = uni["category"].values
n = len(cats_arr)
corr = np.full((n, n), 0.2)
for i in range(n):
    for j in range(n):
        if cats_arr[i] == cats_arr[j]:
            corr[i, j] = 0.7
np.fill_diagonal(corr, 1.0)

# Diversity shrinkage: higher slider → stronger penalty for common risk
adj_cov = base_cov * (1 - diversity_pref * corr)

# =========================================================
# MARKOWITZ ENGINE
# =========================================================
def port_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    r = float(w @ mu)
    v = float(np.sqrt(w.T @ cov @ w))
    return r, v

def min_var(target_r: float, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: float(w @ mu) - target_r},
    )
    res = minimize(lambda w: np.sqrt(w.T @ cov @ w), w0, bounds=bounds, constraints=cons)
    return (res.x if res.success else w0)

# Build efficient frontier on the filtered universe
tgrid = np.linspace(mu.min(), mu.max(), 30)
front = []
for tr in tgrid:
    w = min_var(tr, mu, adj_cov)
    r, v = port_stats(w, mu, adj_cov)
    front.append((r, v))
frontier_df = pd.DataFrame(front, columns=["return", "vol"])

# =========================================================
# TABS: Frontier / Correlations
# =========================================================
tab_frontier, tab_corr = st.tabs(["📉 Frontier & Allocation", "🧩 Correlations & Diversity"])

with tab_frontier:
    # Plot vaults + frontier
    fig = px.scatter(
        uni,
        x=uni["volatility"]*100,
        y=uni["mean_return"]*100,
        color="category",
        color_discrete_map=COLOR_MAP,
        size="tvlUsd",
        hover_data=["name","project","chain","symbol","tvlUsd","observations"],
        labels={"x":"Risk (Volatility %)","y":"Expected Return (APY %)"},
        title="DeFi Vaults — Diversity-Adjusted Efficient Frontier",
    )
    fig.add_scatter(
        x=frontier_df["vol"]*100,
        y=frontier_df["return"]*100,
        mode="lines+markers",
        name="Efficient Frontier",
        line=dict(color="red", width=2),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Universe table
    with st.expander("🧾 Vaults Universe (after filters)", expanded=False):
        st.dataframe(
            uni[["name","project","chain","symbol","category","tvlUsd","mean_return","volatility","observations"]],
            use_container_width=True,
            hide_index=True,
        )

    # Optimal portfolio selection (closest return to target)
    best_idx = (frontier_df["return"]*100 - target_return_pct).abs().idxmin()
    target_r = float(frontier_df.loc[best_idx, "return"])

    # Unconstrained weights
    w_full = min_var(target_r, mu, adj_cov)

    # Cardinality: keep top-K positions, renormalize
    K = int(max_assets)
    order = np.argsort(w_full)[::-1]
    top_idx = order[:K]
    w_card = np.zeros_like(w_full)
    w_card[top_idx] = w_full[top_idx]
    if w_card.sum() <= 0:
        w_card[top_idx] = 1.0 / K
    w_card = w_card / w_card.sum()

    # Final stats + allocation
    port_r, port_v = port_stats(w_card, mu, adj_cov)
    alloc = w_card * float(total_invest)

    out = uni.copy()
    out["Weight"] = w_card
    out["Allocation (USD)"] = alloc
    out = out[out["Weight"] > 0.001].sort_values("Weight", ascending=False)

    st.subheader("💰 Optimal Portfolio Composition (Cardinality-Constrained, Diversity-Adjusted)")
    st.dataframe(
        out[["name","project","category","Weight","Allocation (USD)","mean_return","volatility","tvlUsd"]]
          .rename(columns={"name":"Vault","mean_return":"Expected Return","volatility":"Risk (Volatility)"}),
        use_container_width=True, hide_index=True
    )

    st.markdown(f"""
### 🧭 Portfolio Summary
- **Total Investment:** ${total_invest:,.0f}  
- **Target Return (input):** {target_return_pct:.2f}%  
- **Optimized Return:** {port_r*100:.2f}%  
- **Volatility:** {port_v*100:.2f}%  
- **Selected Vaults:** {(out["Weight"]>0.001).sum()} / {K}  
- **Diversity Preference:** {diversity_pref:.2f}  
- **Last Updated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
""")

with tab_corr:
    st.subheader("Correlation Model (category-aware, simulated)")
    # Show the correlation matrix used (category intra 0.7 / inter 0.2)
    corr_df = pd.DataFrame(corr, index=uni["name"], columns=uni["name"])
    heat = px.imshow(
        corr_df, aspect="auto", color_continuous_scale="RdBu", origin="lower",
        title="Simulated Correlation Matrix (by Category)"
    )
    st.plotly_chart(heat, use_container_width=True)

    # Diversity index: entropy of weights + mean correlation
    # (weights from current optimized portfolio)
    w_safe = w_card.copy()
    w_safe = w_safe[w_safe > 0]
    entropy = -np.sum(w_safe * np.log(w_safe)) / np.log(len(w_card))  # 0..1 normalized
    # mean pairwise corr weighted by weights
    W = np.outer(w_card, w_card)
    mean_corr = float((corr * W).sum())
    st.markdown(f"""
**Diversity Index**
- **Entropy (0–1):** `{entropy:.3f}`  
- **Weighted Mean Correlation:** `{mean_corr:.3f}`  
- **Interpretation:** higher entropy & lower mean corr = better diversification.
""")

st.caption(
    "Notes: When history is unavailable, the app uses DefiLlama snapshot APY and a small volatility proxy from 30-day drift. "
    "Covariance is diagonal scaled by a category-aware correlation model and the Diversity Preference slider. "
    "Cardinality is enforced by keeping top-K weights at the chosen frontier return and renormalizing."
)

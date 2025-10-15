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

API_POOLS = "https://yields.llama.fi/pools"
API_CHART = "https://yields.llama.fi/chart/"
MAX_VAULTS = 80

# =========================================================
# FETCH DATA
# =========================================================
@st.cache_data(show_spinner=True)
def fetch_pools(limit=MAX_VAULTS):
    r = requests.get(API_POOLS, timeout=15)
    r.raise_for_status()
    data = r.json().get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df
    keep = ["pool","project","chain","symbol","tvlUsd","apy","apyMean30d","stablecoin","ilRisk"]
    df = df[keep].copy()
    df = df.sort_values("tvlUsd", ascending=False).head(limit)
    df["apy"] = pd.to_numeric(df["apy"], errors="coerce") / 100
    df["apyMean30d"] = pd.to_numeric(df["apyMean30d"], errors="coerce") / 100
    df["name"] = df.apply(lambda r: f"{r['project']}:{r['symbol']} ({r['chain']})", axis=1)
    return df.dropna(subset=["apy"]).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def fetch_history(pool_id):
    url = f"{API_CHART}{pool_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            return None
        js = r.json()
        if "data" not in js or len(js["data"]) == 0:
            return None
        df = pd.DataFrame(js["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["apy"] = pd.to_numeric(df["apy"], errors="coerce") / 100
        df = df.dropna(subset=["apy"])
        return df if len(df) >= 3 else None
    except Exception:
        return None

@st.cache_data(show_spinner=True)
def build_hybrid_metrics(snapshot):
    rows, skipped = [], []
    progress = st.progress(0.0)
    n = len(snapshot)
    for i, row in snapshot.iterrows():
        pid = row["pool"]
        hist = fetch_history(pid)
        if hist is not None:
            mean_ret = hist["apy"].mean()
            vol = hist["apy"].std()
        else:
            mean_ret = row["apy"]
            vol = abs(row["apy"] - row["apyMean30d"]) if not pd.isna(row["apyMean30d"]) else 0.01
            skipped.append(pid)
        rows.append({
            "pool": pid,
            "name": row["name"],
            "project": row["project"],
            "chain": row["chain"],
            "symbol": row["symbol"],
            "tvlUsd": row["tvlUsd"],
            "stablecoin": row["stablecoin"],
            "ilRisk": row["ilRisk"],
            "mean_return": float(mean_ret),
            "volatility": float(max(vol, 0.0001))
        })
        progress.progress((i+1)/n)
        time.sleep(0.02)
    return pd.DataFrame(rows), skipped

# =========================================================
# CLASSIFICATION
# =========================================================
def classify(row):
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
# LOAD DATA
# =========================================================
snap = fetch_pools(MAX_VAULTS)
st.info("⏳ Building metrics (historical or snapshot fallback)…")
data, skipped_ids = build_hybrid_metrics(snap)
data["category"] = data.apply(classify, axis=1)
data["color"] = data["category"].map(COLOR_MAP)
st.success(f"Loaded {len(data)} vaults — {len(data)-len(skipped_ids)} with history.")

# Cap APY and vol outliers
data["mean_return"] = np.clip(data["mean_return"], 0, 0.5)
data["volatility"] = np.clip(data["volatility"], 0, 0.3)

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
with st.sidebar:
    st.header("⚙️ Interactive Controls")
    cats = sorted(data["category"].unique().tolist())
    selected_cats = st.multiselect("Filter by Category", cats, default=cats, help="Filter vault categories to include.")
    uni = data[data["category"].isin(selected_cats)].copy()

    chosen = st.multiselect(
        "Vault Universe (optional)",
        options=uni["name"].tolist(),
        default=[],
        help="Select specific vaults to focus on (leave empty for all)."
    )
    if chosen:
        uni = uni[uni["name"].isin(chosen)].copy()

    target_return_pct = st.slider("🎯 Target Return (APY %)", 0.0, float(max(0.5, uni["mean_return"].max()*100)), 10.0)
    diversity_pref = st.slider("🌈 Diversity Preference", 0.0, 1.0, 0.5, help="Adjust correlation weight between vaults.")
    max_assets = st.slider("📦 Max Assets", 3, min(25, max(3,len(uni))), 8)
    total_invest = st.number_input("💵 Total Investment (USD)", min_value=1000, max_value=1_000_000, value=50_000)
    rf = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)

# =========================================================
# DIVERSITY-AWARE COVARIANCE
# =========================================================
mu = uni["mean_return"].values
sigma = uni["volatility"].values
base_cov = np.diag(sigma**2)
cats_arr = uni["category"].values
n = len(cats_arr)
corr = np.full((n,n),0.2)
for i in range(n):
    for j in range(n):
        if cats_arr[i]==cats_arr[j]:
            corr[i,j]=0.7
np.fill_diagonal(corr,1)
adj_cov = base_cov * (1 - diversity_pref * corr)

# =========================================================
# MARKOWITZ FUNCTIONS
# =========================================================
def port_stats(w, mu, cov):
    r = w @ mu
    v = np.sqrt(w.T @ cov @ w)
    return r, v

def min_var(target_r, mu, cov):
    n=len(mu)
    w0=np.ones(n)/n
    b=[(0,1)]*n
    c=(
        {"type":"eq","fun":lambda w: np.sum(w)-1},
        {"type":"eq","fun":lambda w: w@mu - target_r}
    )
    res=minimize(lambda w: np.sqrt(w.T@cov@w), w0, bounds=b, constraints=c)
    return res.x if res.success else w0

# =========================================================
# FRONTIER
# =========================================================
tgrid=np.linspace(mu.min(),mu.max(),40)
front=[]
for tr in tgrid:
    w=min_var(tr,mu,adj_cov)
    r,v=port_stats(w,mu,adj_cov)
    front.append((r,v))
frontier_df=pd.DataFrame(front,columns=["return","vol"])

# =========================================================
# RANDOM PORTFOLIO SIMULATION
# =========================================================
N=1500
rw=np.random.dirichlet(np.ones(len(mu)),size=N)
r_ret=rw@mu
r_vol=np.sqrt(np.einsum('ij,jk,ik->i',rw,adj_cov,rw))
rand_df=pd.DataFrame({"return":r_ret*100,"volatility":r_vol*100})

# =========================================================
# CAPITAL ALLOCATION LINE (CAL)
# =========================================================
rf_dec = rf / 100
sharpe=(frontier_df["return"]-rf_dec)/frontier_df["vol"]
tangent_idx=sharpe.idxmax()
tangent=frontier_df.loc[tangent_idx]
x_line=np.linspace(0,frontier_df["vol"].max(),100)
y_line=rf_dec + (tangent["return"]-rf_dec)/tangent["vol"]*x_line

# =========================================================
# MAIN TAB LAYOUT
# =========================================================
tab1, tab2, tab3 = st.tabs(["📉 Frontier & Portfolio", "🎲 Random Portfolios", "🧩 Correlations & Diversity"])

# --- FRONTIER ---
with tab1:
    fig=px.scatter(
        uni,
        x=uni["volatility"]*100,
        y=uni["mean_return"]*100,
        color="category",
        color_discrete_map=COLOR_MAP,
        size="tvlUsd",
        hover_data=["name","project","chain","symbol","tvlUsd"],
        labels={"x":"Risk (Volatility %)","y":"Expected Return (APY %)"},
        title="DeFi Vaults — Efficient Frontier with CAL"
    )

    fig.add_scatter(x=frontier_df["vol"]*100,y=frontier_df["return"]*100,
                    mode="lines+markers",name="Efficient Frontier",
                    line=dict(color="red",width=2))
    fig.add_scatter(x=x_line*100,y=y_line*100,mode="lines",
                    name="Capital Allocation Line (CAL)",
                    line=dict(color="black",width=3))
    st.plotly_chart(fig,use_container_width=True)

    # Vault Universe table
    with st.expander("🧾 Vault Universe (Full List)", expanded=False):
        st.dataframe(
            uni[["name","project","chain","symbol","category","tvlUsd","mean_return","volatility"]],
            use_container_width=True,hide_index=True
        )

    # Optimal Portfolio
    best_idx=(frontier_df["return"]*100 - target_return_pct).abs().idxmin()
    target_r=frontier_df.loc[best_idx,"return"]
    w_full=min_var(target_r,mu,adj_cov)
    K=int(max_assets)
    top_idx=np.argsort(w_full)[::-1][:K]
    w_card=np.zeros_like(w_full)
    w_card[top_idx]=w_full[top_idx]
    w_card=w_card/w_card.sum()
    port_r,port_v=port_stats(w_card,mu,adj_cov)
    alloc=w_card*total_invest
    out=uni.copy()
    out["Weight"]=w_card
    out["Allocation (USD)"]=alloc
    out=out[out["Weight"]>0.001].sort_values("Weight",ascending=False)
    tangent_sharpe=(tangent["return"]-rf_dec)/tangent["vol"]

    st.subheader("💰 Optimal Portfolio")
    st.dataframe(
        out[["name","project","category","Weight","Allocation (USD)","mean_return","volatility"]],
        use_container_width=True,hide_index=True
    )

    st.markdown(f"""
### 🧭 Portfolio Summary
- **Total Investment:** ${total_invest:,.0f}
- **Target Return (input):** {target_return_pct:.2f}%
- **Optimized Return:** {port_r*100:.2f}%
- **Volatility:** {port_v*100:.2f}%
- **Sharpe (tangent):** {tangent_sharpe:.2f}
- **Vaults Selected:** {(out["Weight"]>0.001).sum()} / {K}
- **Diversity Preference:** {diversity_pref:.2f}
- **Last Updated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
""")

# --- RANDOM PORTFOLIO SIMULATION ---
with tab2:
    fig_rand=px.scatter(
        rand_df,x="volatility",y="return",
        title="Random Portfolios Simulation — Risk vs Return",
        labels={"volatility":"Risk (Volatility %)","return":"Expected Return (APY %)"},
        opacity=0.6,color_discrete_sequence=["orange"]
    )
    fig_rand.add_scatter(
        x=frontier_df["vol"]*100,y=frontier_df["return"]*100,
        mode="lines",name="Efficient Frontier",line=dict(color="red",width=2)
    )
    st.plotly_chart(fig_rand,use_container_width=True)
    st.caption("Orange points = randomly generated portfolio allocations. The red curve = efficient frontier.")

# --- CORRELATION TAB ---
with tab3:
    corr_df=pd.DataFrame(corr,index=uni["name"],columns=uni["name"])
    heat=px.imshow(corr_df,aspect="auto",color_continuous_scale="RdBu",origin="lower")
    st.plotly_chart(heat,use_container_width=True)
    w_safe=w_card[w_card>0]
    entropy=-np.sum(w_safe*np.log(w_safe))/np.log(len(w_card))
    W=np.outer(w_card,w_card)
    mean_corr=float((corr*W).sum())
    st.markdown(f"""
**Diversity Index**
- Entropy (0–1): `{entropy:.3f}`
- Weighted Mean Correlation: `{mean_corr:.3f}`
- Interpretation: higher entropy & lower mean corr = better diversification.
""")

st.caption(
    "Notes: The orange points represent random portfolios, "
    "the red curve is the efficient frontier, and the black line is the Capital Allocation Line (CAL). "
    "All APYs > 50% are capped to ensure realistic scales."
)

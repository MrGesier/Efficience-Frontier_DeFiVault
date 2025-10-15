# 📈 DeFi Vaults Efficient Frontier
**by Mr_Gesier | Nomiks**

A Streamlit-based analytics tool that visualizes and optimizes DeFi yield vaults through a **Markowitz efficient frontier** approach.  
It aggregates live yield data from [DefiLlama](https://defillama.com), classifies vaults by category, and provides an interactive view of **risk, return, and diversification**.

---

## 🚀 Features

- 🔗 **Live data** from [DefiLlama API](https://yields.llama.fi/)
- 🧠 **Efficient frontier** simulation (Markowitz optimizer)
- 🎨 **Category-based color coding** (Stablecoin / BTC / LSD / Credit / LP)
- ⚙️ **Interactive sliders** for risk tolerance, target return, and diversity weighting
- 🪙 **Vault filtering** by taxonomy (via sidebar)
- 📊 **Dynamic Plotly chart** + **vault list table**
- 🧾 **Portfolio summary** (expected return, volatility, diversity)

---

## 🧩 Example Layout

**Main view:**
- Scatter plot of vaults → x-axis = volatility, y-axis = expected return (APY)
- Red line = efficient frontier (optimal risk-return curve)
- Tooltip shows: protocol, chain, symbol, APY, 30d APY, and TVL

**Sidebar:**
- Risk tolerance slider  
- Target return slider  
- Diversity weight slider  
- Category multiselect (Stablecoin / BTC / LSD / Structured / LP)

---

## 🧰 Installation

### 1️⃣ Clone this repository
```bash
git clone https://github.com/yourusername/defi-frontier.git
cd defi-frontier

<img width="1864" height="864" alt="image" src="https://github.com/user-attachments/assets/fc7e08c2-011c-4c24-aac4-04b28a173f59" />
<img width="1864" height="890" alt="image" src="https://github.com/user-attachments/assets/0d9ae5ae-315d-4f48-89f4-78af356b22a2" />

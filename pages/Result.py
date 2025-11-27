import streamlit as st
import json
import pandas as pd
from data.db import load_simulations, delete_simulation, clear_database

st.set_page_config(page_title="Simulation Results", layout="wide")
st.title("📊 Delegated Lending Simulation Results")

# =========================================================
# Load stored simulations
# =========================================================
df = load_simulations()

if df is None or df.empty:
    st.warning("No simulations found. Run a simulation first.")
    st.stop()

# =========================================================
# Simulation chooser
# =========================================================
st.markdown("### 📁 Select a simulation run")

selected_row = st.selectbox(
    "Simulation history",
    df.index,
    format_func=lambda idx: f"{df.loc[idx, 'sim_name']} — {df.loc[idx, 'run_time']}"
)

sim = df.loc[selected_row]

if "results" not in sim or sim["results"] is None:
    st.error("This simulation has no stored results. Run a new simulation.")
    st.stop()

# Load results JSON
results = sim["results"]
if isinstance(results, str):
    results = json.loads(results)

st.divider()

# =========================================================
# Overview Header
# =========================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Simulation Name", sim["sim_name"])
col2.metric("Number of Firms", sim["num_firms"])
col3.metric("Principal's Expected Welfare", f"{sim['welfare']:.4f}")
col4.metric("Iterations", sim["iterations"])

st.divider()

# =========================================================
# MARKET SUMMARY (NEW)
# =========================================================
st.markdown("## 📊 Market Summary Dashboard")

# Build analysis dataframe
summary_df = []
for firm in results:
    choice = firm["choice"]

    summary_df.append({
        "firm_id": firm["firm_id"],
        "true_type": choice["true_type"],
        "intended": choice["principal_intended_plan"],
        "chosen": choice["chosen_plan"],
        "deviation": choice["deviation"],
        "non_participation": choice["non_participation"],
        "payoff": choice["chosen_payoff"],
    })

summary_df = pd.DataFrame(summary_df)

# Summary metrics
num_firms = len(summary_df)
num_opt_out = summary_df["non_participation"].sum()
num_matched = (summary_df["chosen"] == summary_df["intended"]).sum()
num_deviations = summary_df["deviation"].sum()

match_rate = num_matched / num_firms
deviation_rate = num_deviations / num_firms
opt_out_rate = num_opt_out / num_firms
avg_payoff = summary_df["payoff"].mean()

# Quick Status Panel
colA, colB, colC, colD = st.columns(4)
colA.metric("Match Rate", f"{match_rate*100:.1f}%")
colB.metric("Deviation Rate", f"{deviation_rate*100:.1f}%")
colC.metric("Opt-out Rate", f"{opt_out_rate*100:.1f}%")
colD.metric("Avg Realized Payoff", f"{avg_payoff:.3f}")

# Insights
if opt_out_rate > 0.3:
    st.warning("🔴 Many firms opted out — principal’s catalog may be unattractive.")

elif deviation_rate > 0.5:
    st.warning("🟡 High deviation rate — firms are not choosing their intended plan.")

else:
    st.success("🟢 Catalog is reasonably effective.")

st.divider()

# =========================================================
#  VISUALIZATIONS (NEW)
# =========================================================
st.markdown("## 📈 Visualizations")

# ---------- CHART 1: Chosen Plan Distribution ----------
st.markdown("### 📌 Chosen Plans Distribution")

chosen_counts = summary_df["chosen"].value_counts().reset_index()
chosen_counts.columns = ["Plan", "Count"]

st.bar_chart(chosen_counts.set_index("Plan"))

colA , colB = st.columns(2)

with colA:
    # ---------- CHART 2: Participation Split ----------
    st.markdown("### 🥧 Participation vs Non-Participation")

    participation_df = pd.DataFrame({
        "category": ["Participating", "Opt-Out"],
        "count": [num_firms - num_opt_out, num_opt_out]
    })

    st.bar_chart(participation_df.set_index("category"))
    
with colB:
    # ---------- CHART 3: Deviations ----------
    st.markdown("### 🔄 Deviations (Matched vs Deviated)")

    dev_df = pd.DataFrame({
        "category": ["Matched", "Deviated"],
        "count": [num_matched, num_deviations]
    })

    st.bar_chart(dev_df.set_index("category"))
    
st.divider()

# =========================================================
#  PER-FIRM DETAILS (your existing section)
# =========================================================
st.markdown("## 🏢 Firm-by-Firm Details")

for firm in results:
    firm_id = firm["firm_id"]
    principal = firm["principal"]
    plans = firm["plans"]
    choice = firm["choice"]

    st.markdown(f"### Firm {firm_id}")

    # Beliefs
    with st.expander("🎯 Principal Beliefs"):
        belief_df = pd.DataFrame({
            "Metric": ["Belief SAFE (%)", "Safe", "Mixed", "Risky"],
            "Value": [
                f"{principal['belief_safe_score']:.1f}%",
                f"{principal['pi_safe']:.3f}",
                f"{principal['pi_mixed']:.3f}",
                f"{principal['pi_risky']:.3f}",
            ]
        })
        st.table(belief_df)

    # Catalog Plans
    with st.expander("📜 Catalog Plans (Safe / Mixed / Risky)"):
        for plan_name, plan_data in plans.items():
            st.markdown(f"#### Plan: {plan_name.capitalize()}")
            cols = st.columns(2)
            with cols[0]:
                st.write("**Club A (Short-Term)**")
                st.json(plan_data["club_A"])
            with cols[1]:
                st.write("**Club B (Long-Term)**")
                st.json(plan_data["club_B"])
            st.divider()

    # Firm Decision
    with st.expander("🤖 Firm Decision"):
        st.write(f"**True Type:** `{choice['true_type']}`")
        st.write(f"**Intended Plan:** `{choice['principal_intended_plan']}`")
        st.write(f"**Chosen Plan:** `{choice['chosen_plan']}`")
        st.write(f"**Payoff:** {choice['chosen_payoff']:.3f}")

        status = "🟢 Matched" if not choice["deviation"] else "🟡 Deviated"
        if choice["non_participation"]:
            status = "🔴 Opt-Out (Club C)"

        st.markdown(f"### Status: {status}")

        payoff_table = pd.DataFrame({
            "Plan": ["Safe", "Mixed", "Risky", "Opt-Out"],
            "Payoff": [
                choice["payoffs"]["safe"],
                choice["payoffs"]["mixed"],
                choice["payoffs"]["risky"],
                choice["payoffs"]["opt_out"],
            ]
        })
        st.dataframe(payoff_table, hide_index=True, use_container_width=True)

    st.divider()

# =========================================================
# Delete Controls
# =========================================================
with st.expander("🗑 Delete Simulations"):
    colA, colB = st.columns(2)
    with colA:
        if st.button("Delete Selected Simulation"):
            delete_simulation(int(sim["id"]))
            st.success("Simulation deleted.")
            st.experimental_rerun()
    with colB:
        if st.button("Clear ALL Simulations"):
            clear_database()
            st.success("All simulations deleted.")
            st.experimental_rerun()
